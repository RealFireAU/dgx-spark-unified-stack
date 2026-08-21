#!/usr/bin/env python3
"""Compute LiteLLM model_info cost fields (input/output_cost_per_token) from
real hardware ownership costs, so "local and free" isn't treated as literally
$0 — depreciation and electricity are real costs even for owned hardware.

Methodology (see cost-model.md in this directory for the full writeup):
    hourly_cost = depreciation_per_hour + electricity_per_hour
    depreciation_per_hour = hardware_cost / (depreciation_years * 365.25 * 24)
    electricity_per_hour  = (active_power_watts / 1000) * electricity_rate_per_kwh
    cost_per_token        = (hourly_cost / 3600) / measured_tokens_per_second

Currency: whatever you pass in is what comes out — LiteLLM's cost map is
USD-denominated by convention, so if you pass AUD figures (as this repo
does, hardware_cost_aud default), the resulting numbers ARE in AUD even
though LiteLLM's dashboard will show them with a "$" that normally implies
USD. Documented, not converted — exchange rates drift, real hardware/power
costs don't need one more unverified assumption layered in.

Usage:
    python3 estimate_model_cost.py --tps 79.68 --prompt-tps 53.19
    python3 estimate_model_cost.py --tps 17.33 --hardware-cost 9000 \
        --depreciation-years 3 --power-watts 170 --electricity-rate 0.30
"""
import argparse


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tps", type=float, required=True, help="measured output tokens/sec for this model")
    p.add_argument("--prompt-tps", type=float, default=None, help="measured prompt/input tokens/sec (defaults to --tps if unset)")
    p.add_argument("--hardware-cost", type=float, default=9000.0, help="hardware purchase cost, default AUD 9000 (DGX Spark)")
    p.add_argument("--depreciation-years", type=float, default=3.0, help="straight-line depreciation period, default 3 years")
    p.add_argument("--power-watts", type=float, default=170.0, help="active inference power draw estimate, default 170W (140W GB10 TDP + ~30W system overhead; NVIDIA's spec: 240W PSU max, 140W GB10 chip TDP)")
    p.add_argument("--electricity-rate", type=float, default=0.2955, help="electricity cost per kWh, default 0.2955 AUD (real plan rate).")
    args = p.parse_args()

    prompt_tps = args.prompt_tps if args.prompt_tps is not None else args.tps

    depreciation_per_hour = args.hardware_cost / (args.depreciation_years * 365.25 * 24)
    electricity_per_hour = (args.power_watts / 1000) * args.electricity_rate
    hourly_cost = depreciation_per_hour + electricity_per_hour

    output_cost_per_token = (hourly_cost / 3600) / args.tps
    input_cost_per_token = (hourly_cost / 3600) / prompt_tps

    print(f"depreciation_per_hour: {depreciation_per_hour:.6f}")
    print(f"electricity_per_hour:  {electricity_per_hour:.6f}")
    print(f"hourly_cost:           {hourly_cost:.6f}")
    print()
    print(f"output_cost_per_token: {output_cost_per_token:.10f}")
    print(f"input_cost_per_token:  {input_cost_per_token:.10f}")
    print()
    print("model_info:")
    print(f"  input_cost_per_token: {input_cost_per_token:.10f}")
    print(f"  output_cost_per_token: {output_cost_per_token:.10f}")


if __name__ == "__main__":
    main()
