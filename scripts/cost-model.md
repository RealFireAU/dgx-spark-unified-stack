# Model cost model

"Local and free" isn't literally $0 — the hardware cost real money and
running it costs real electricity. This estimates a per-token cost for
each model, so LiteLLM's cost tracking reflects that instead of showing 0.

Cost fields live in each model's own `metadata:` block under
`models/<name>.yaml`, not hand-written into `LiteLLM/config.yaml` —
`scripts/generate_litellm_models.py` copies them into LiteLLM's
`model_info` automatically. (LiteLLM's `model_list` is static config — it
does not live-query an upstream OpenAI-compatible provider for pricing, so
there's no way to make this "just work" without setting it explicitly
somewhere.)

## Inputs

- **Hardware cost**: your purchase price for the machine.
- **Depreciation period**: straight-line, calendar-based (the asset ages
  whether it's running or not — the simplest defensible accounting
  approach, not the only valid one) — pick a period matching how long you
  expect to actually use it.
- **Power draw while actively generating**: use the chip's spec'd TDP plus
  a modest estimate for system overhead (storage, fans, networking) unless
  you have a real wall-meter measurement. Idle power is lower — this model
  only prices *active* generation time, not idle sitting time.
- **Electricity rate**: your real per-kWh plan rate.

## Formula

```
depreciation_per_hour = hardware_cost / (depreciation_years * 365.25 * 24)
electricity_per_hour  = (power_watts / 1000) * electricity_rate
hourly_cost           = depreciation_per_hour + electricity_per_hour

cost_per_token         = (hourly_cost / 3600) / measured_tokens_per_second
```

**Worked example** (illustrative numbers, not this deployment's real
figures):

```
hardware_cost = 9000, depreciation_years = 3, power_watts = 170, electricity_rate = 0.30

depreciation_per_hour = 9000 / (3 * 365.25 * 24)  = $0.3422/hour
electricity_per_hour  = 0.170 * 0.30              = $0.0510/hour
hourly_cost           = $0.3932/hour

# at 20 output tok/s:
output_cost_per_token = (0.3932 / 3600) / 20 = $0.00000546
```

Input (prompt) and output (generation) tokens are priced separately using
each model's *own measured* prompt-processing and generation throughput —
prompt processing is typically much faster, so input tokens are cheaper.

**Currency note**: LiteLLM's `input_cost_per_token`/`output_cost_per_token`
are USD-denominated by convention, but nothing stops you feeding in
whatever currency your real hardware/electricity costs are in — no FX
conversion is applied. Read the `$` in LiteLLM's UI as your own currency,
not necessarily USD.

## Measuring real throughput, not guessing

Extrapolating a model's throughput from parameter count (or from another
model's measured number) produces precise-looking numbers that are
actually guesses. Only give a model a `metadata:` cost block once you have
a real measured number for it — leave the rest unpriced rather than
inventing a plausible-looking one.

```bash
python3 scripts/estimate_model_cost.py --tps <measured_output_tps> --prompt-tps <measured_prompt_tps>
```

Boot the model, send a real generation request, read
`timings.predicted_per_second` and `timings.prompt_per_second` from
llama.cpp's response (vLLM's OpenAI-format response doesn't include this —
time the request wall-clock and divide by
`usage.completion_tokens`/`usage.prompt_tokens` instead), then feed those
into the script above and add the output as a `metadata:` block on that
model's `models/<name>.yaml` file (see `models.example/example-vllm-model.yaml`
for the pattern).
