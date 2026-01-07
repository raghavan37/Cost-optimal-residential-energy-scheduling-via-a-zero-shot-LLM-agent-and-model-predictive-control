!pip install pvlib
!pip install requests beautifulsoup4 lxml
!pip install liionpack pybamm

import pybamm
import liionpack as lp
import numpy as np
import pvlib
import pandas as pd
import json
import re
import requests
from bs4 import BeautifulSoup
import datetime
import os
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import copy
from scipy.optimize import minimize
import time
from sklearn.metrics import f1_score
from functools import partial

# --- NEW: Import SDKs for different models ---
import google.generativeai as genai
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# --- Configuration for Different LLM APIs ---

# 1. Gemini API Configuration
SYSTEM_PROMPT_GEMINI = """
You are an expert system that translates a user's natural language query into a structured JSON tool call.
Your response MUST be a single JSON object with two keys: "tool_name" and "parameters".
The "tool_name" must be one of the following: ["mpc_simulation", "baseline_reactive_price_blind", "heuristic_price_aware_baseline", "get_solar_forecast", "plot_solar_forecast", "get_electricity_price", "plot_electricity_price", "control_appliance", "suggest_optimal_appliance_time_with_mpc", "N/A"].
Use "N/A" if the user's query is out of scope, a greeting, or nonsensical.
The "parameters" must be a JSON object containing the extracted parameters for the tool. All date parameters MUST use the key "date".

Tool Descriptions:
- mpc_simulation: Use for general energy planning, optimization, or scheduling for a full day. Triggers on "plan energy", "optimize schedule", "run mpc". Use this for general "optimization" queries that do NOT name a specific appliance.
- baseline_reactive_price_blind: Use for generating a basic, non-optimized, price-blind energy plan. Triggers on "baseline plan", "basic simulation", "price-blind plan".
- heuristic_price_aware_baseline: Use for generating a rule-based, price-aware energy plan (better than baseline, simpler than MPC). Triggers on "heuristic plan", "rule-based plan", "price-aware baseline".
- get_solar_forecast: Get numerical solar data.
- plot_solar_forecast: Create a visual plot of solar data.
- get_electricity_price: Get numerical electricity price data.
- plot_electricity_price: Create a visual plot of electricity price data.
- control_appliance: Directly control an appliance (on/off/schedule).
- suggest_optimal_appliance_time_with_mpc: Use ONLY when a specific appliance NAME (e.g., "Washing Machine", "Geyser") is mentioned in the query. Triggers on "when should I run", "best time for". If no appliance is named, DO NOT use this tool.

Today's date is 2024-09-27.
"""
try:
    GEMINI_API_KEY = "AIzaSyBwaCh3ap8QXA-YV-VoJghKsrlA3g1iN54" # This will be automatically configured in many environments
    genai.configure(api_key=GEMINI_API_KEY)
    llm_model_gemini = genai.GenerativeModel(
        'gemini-2.5-flash',
        system_instruction=SYSTEM_PROMPT_GEMINI
    )
    print("Gemini 2.5 Flash model configured successfully.")
except Exception as e:
    print(f"Could not configure Gemini model: {e}")
    llm_model_gemini = None

# # 2. Ollama API Configuration
# SYSTEM_PROMPT_OLLAMA = """
# You are an expert system that translates a user's natural language query into a structured JSON tool call.
# Your response MUST be a single JSON object with two keys: "tool_name" and "parameters".
# The "tool_name" must be one of the following: ["mpc_simulation", "baseline_reactive_price_blind", "heuristic_price_aware_baseline", "get_solar_forecast", "plot_solar_forecast", "get_electricity_price", "plot_electricity_price", "control_appliance", "suggest_optimal_appliance_time_with_mpc", "N/A"].
# Use "N/A" if the user's query is out of scope, a greeting, or nonsensical.
# The "parameters" must be a JSON object containing the extracted parameters for the tool. All date parameters MUST use the key "date".

# Tool Descriptions:
# - mpc_simulation: Use for general energy planning, optimization, or scheduling for a full day. Triggers on "plan energy", "optimize schedule", "run mpc". Use this for general "optimization" queries that do NOT name a specific appliance.
# - baseline_reactive_price_blind: Use for generating a basic, non-optimized, price-blind energy plan. Triggers on "baseline plan", "basic simulation", "price-blind plan".
# - heuristic_price_aware_baseline: Use for generating a rule-based, price-aware energy plan (better than baseline, simpler than MPC). Triggers on "heuristic plan", "rule-based plan", "price-aware baseline".
# - get_solar_forecast: Get numerical solar data.
# - plot_solar_forecast: Create a visual plot of solar data.
# - get_electricity_price: Get numerical electricity price data.
# - plot_electricity_price: Create a visual plot of electricity price data.
# - control_appliance: Directly control an appliance (on/off/schedule).
# - suggest_optimal_appliance_time_with_mpc: Use ONLY when a specific appliance NAME (e.g., "Washing Machine", "Geyser") is mentioned in the query. Triggers on "when should I run", "best time for". If no appliance is named, DO NOT use this tool.
# """
# ollama_client = None
# if OLLAMA_AVAILABLE:
#     try:
#         # By default, the client connects to http://localhost:11434, which is correct for a Colab setup.
#         ollama_client = ollama.Client()
#         # Quick check to see if the server is responsive
#         ollama_client.list()
#         print("Successfully connected to local Ollama server running in Colab.")
#     except Exception as e:
#         print("Could not connect to local Ollama server. Please ensure it's running in your Colab environment using the setup instructions.")
#         print(f"Error details: {e}")
#         ollama_client = None

# --- Modular API Callers for Different Models ---

def call_gemini_for_tool(user_query: str) -> Dict:
    """ Calls the Gemini Flash API via the SDK. """
    if not llm_model_gemini:
        return {"tool_name": "N/A", "parameters": {"error": "LLM not configured"}}

    generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
    try:
        response = llm_model_gemini.generate_content(user_query, generation_config=generation_config)
        parsed_json = json.loads(response.text)
        return {"tool_name": parsed_json.get("tool_name", "N/A"), "parameters": parsed_json.get("parameters", {})}
    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        return {"tool_name": "N/A", "parameters": {"error": f"API call failed: {e}"}}

# def call_ollama_for_tool(user_query: str, model_name: str = 'llama3:8b-instruct') -> Dict:
#     """ Calls a local Ollama model for tool selection. """
#     if not OLLAMA_AVAILABLE:
#         raise ConnectionError("The 'ollama' library is not installed. Please run 'pip install ollama'.")

#     try:
#         response = ollama.chat(
#             model=model_name,
#             messages=[
#                 {'role': 'system', 'content': SYSTEM_PROMPT_OLLAMA},
#                 {'role': 'user', 'content': user_query},
#             ],
#             options={'temperature': 0.0},
#             format='json'
#         )

#         parsed_json = json.loads(response['message']['content'])
#         return {"tool_name": parsed_json.get("tool_name", "N/A"), "parameters": parsed_json.get("parameters", {})}
#     except json.JSONDecodeError as e:
#         print(f"Error decoding JSON from Ollama for query '{user_query}': {e}")
#         print(f"Raw Ollama response: {response['message']['content']}")
#         return {"tool_name": "N/A", "parameters": {"error": "JSONDecodeError"}}
#     except Exception as e:
#         print(f"Error during Ollama API call: {e}")
#         raise e


# --- IMPORTANT: Placeholder definitions for missing variables ---
params = {
    'soc_init': 0.5,
    'Ns': 60,
    'Np': 10,
    'V_cell_nominal': 3.65
}

# Global definition of default appliances data for consistent access
DEFAULT_APPLIANCES_DATA = [
    # --- Heavy Chores ---
    {
        "name": "Water Pump (1hp)", "type": "Controllable", "power": 746, "duration": 1.5,
        "schedule_weekday": [{"start": 6.5, "end": 9}],
        "schedule_weekend": [{"start": 7, "end": 10}]
    },
    {
        "name": "Washing Machine", "type": "Controllable", "power": 1800, "duration": 1,
        "schedule_weekday": [{"start": 6.5, "end": 9}],
        "schedule_weekend": [{"start": 8, "end": 18}]
    },
    {
        "name": "Dish Washer", "type": "Controllable", "power": 1700, "duration": 1.25,
        "schedule_weekday": [{"start": 21, "end": 23}],
        "schedule_weekend": [{"start": 14, "end": 16}]
    },
    {
        "name": "Iron", "type": "Controllable", "power": 750, "duration": 0.5,
        "schedule_weekday": [{"start": 7, "end": 8.5}],
        "schedule_weekend": [{"start": 10, "end": 13}]
    },
    # --- Climate Control ---
    {
        "name": "AC", "type": "Thermostatically", "power": 1150, "duration": 5,
        "schedule_weekday": [{"start": 0, "end": 5}, {"start": 21, "end": 24}],
        "schedule_weekend": [{"start": 0, "end": 6}, {"start": 14, "end": 17}, {"start": 20, "end": 24}]
    },
    {
        "name": "Geyser (10 l)", "type": "Controllable", "power": 2000, "duration": 0.5,
        "schedule_weekday": [{"start": 5.5, "end": 8}, {"start": 18.5, "end": 21}],
        "schedule_weekend": [{"start": 6.5, "end": 10}, {"start": 19, "end": 21}]
    },
    {
        "name": "Refrigerator", "type": "Thermostatically", "power": 87.5, "duration": 24,
        "schedule_weekday": [{"start": 0, "end": 24}]
    },
    {
        "name": "Electrical Cooker", "type": "Controllable", "power": 559.5, "duration": 0.75,
        "schedule_weekday": [{"start": 6.5, "end": 8.5}, {"start": 19, "end": 20.5}],
        "schedule_weekend": [{"start": 8, "end": 10}, {"start": 13, "end": 15}, {"start": 19, "end": 21}]
    },
    {
        "name": "Microwave Oven", "type": "Controllable", "power": 1200, "duration": 0.1,
        "schedule_weekday": [{"start": 7, "end": 9}, {"start": 12, "end": 14}, {"start": 19, "end": 21}],
        "schedule_weekend": [{"start": 8, "end": 22}]
    },
    {
        "name": "Mixer/Grinder", "type": "Controllable", "power": 600, "duration": 0.15,
        "schedule_weekday": [{"start": 7, "end": 8.5}, {"start": 19, "end": 20}],
        "schedule_weekend": [{"start": 8.5, "end": 10}, {"start": 13, "end": 14.5}]
    },
    {
        "name": "Water Purifier (RO)", "type": "Uncontrollable", "power": 25, "duration": 3,
        "schedule_weekday": [{"start": 0, "end": 24}]
    },
    # --- Entertainment & Personal Devices ---
    {
        "name": "TV", "type": "Controllable", "power": 200, "duration": 2.5,
        "schedule_weekday": [{"start": 19, "end": 22}],
        "schedule_weekend": [{"start": 14, "end": 23}]
    },
    {
        "name": "Laptop", "type": "Controllable", "power": 100, "duration": 2,
        "schedule_weekday": [{"start": 18, "end": 22}],
        "schedule_weekend": [{"start": 10, "end": 22}]
    },
    {
        "name": "Device Chargers", "type": "Uncontrollable", "power": 20, "duration": 8,
        "schedule_weekday": [{"start": 0, "end": 7}, {"start": 20, "end": 24}],
        "schedule_weekend": [{"start": 0, "end": 24}]
    },
    # --- General Loads ---
    {
        "name": "Fans", "type": "Uncontrollable", "power": 180, "duration": 1,
        "schedule_weekday": [{"start": 0, "end": 8}, {"start": 18, "end": 24}],
        "schedule_weekend": [{"start": 0, "end": 24}]
    },
    {
        "name": "Main Lighting", "type": "Uncontrollable", "power": 100, "duration": 1,
        "schedule_weekday": [{"start": 5, "end": 7}, {"start": 18, "end": 23}],
        "schedule_weekend": [{"start": 5, "end": 7}, {"start": 18, "end": 24}]
    },
    {
        "name": "Ambient Lighting", "type": "Thermostatically", "power": 15,  "duration": 24, "schedule_weekday": [{"start": 0, "end": 24}]
    },
    {"name": "Sensors", "type": "Uncontrollable", "power": 10, "duration": 24, "schedule_weekday": [{"start": 0, "end": 24}]},
]


class BatterySimulator:
    def __init__(self, Ns=60, Np=10, V_cell_nominal=3.65):
        """
        Initializes the BatterySimulator with fixed battery configuration.
        Simulation parameters (power, initial SOC, V_pack for current calc)
        will be passed dynamically to run_single_interval.
        """
        self.Ns = Ns
        self.Np = Np
        self.V_cell_nominal = V_cell_nominal

        self.parameters = pybamm.ParameterValues("Chen2020")
        self.model = pybamm.lithium_ion.SPMe(options={"thermal": "lumped"})
        self.param = self.model.param

        self.x_0, self.x_100, self.y_100, self.y_0 = pybamm.lithium_ion.get_min_max_stoichiometries(self.parameters, self.param)
        self.c_n_max = self.parameters["Maximum concentration in negative electrode [mol.m-3]"]
        self.c_p_max = self.parameters["Maximum concentration in positive electrode [mol.m-3]"]

        self.var_pts = {"x_n": 30, "x_s": 30, "x_p": 30, "r_n": 10, "r_p": 10}

        self.output_vars = [
            "State of Charge",
            "Terminal voltage [V]",
            "Current [A]",
            "Power [W]",
            "Average negative particle concentration",
            "X-averaged positive particle concentration [mol.m-3]"
        ]

        self.netlist = lp.setup_circuit(Np=Np, Ns=Ns, Rb=1e-3, Rc=1e-2, Ri=0.002, V=V_cell_nominal)


    def create_single_experiment_step(self, power_kW, V_pack_for_current_calc):
        """
        Creates a single constant current experiment step for PyBaMM.
        Current is approximated based on the provided power and pack voltage from the previous interval.
        """
        interval_duration_str = "15 minutes"
        power_W = power_kW * 1000

        if V_pack_for_current_calc == 0:
            print("Warning: V_pack_for_current_calc is zero. Using nominal pack voltage for current calculation.")
            V_pack_for_current_calc = self.V_cell_nominal * self.Ns
            if V_pack_for_current_calc == 0:
                raise ValueError("Nominal pack voltage is also zero. Cannot calculate current.")

        if V_pack_for_current_calc == 0 and power_W == 0:
            pack_current_A = 0.0
        elif V_pack_for_current_calc == 0:
            print(f"Warning: Attempting to calculate current with zero pack voltage and non-zero power ({power_W} W). Setting current to max rate.")
            pack_current_A = 1000.0
        else:
            pack_current_A = power_W / V_pack_for_current_calc

        cell_current_A = pack_current_A / self.Np

        return pybamm.Experiment([pybamm.step.current(cell_current_A, duration=interval_duration_str)])


    def run_single_interval_simulation(self, power_kW, initial_soc_for_interval, V_pack_for_current_calc):
        """
        Runs a simulation for a single 15-minute interval.
        """
        experiment = self.create_single_experiment_step(power_kW, V_pack_for_current_calc)

        initial_x_neg = self.x_0 + (self.x_100 - self.x_0) * initial_soc_for_interval
        initial_y_pos_stoich = self.y_0 * (1 - initial_soc_for_interval) + self.y_100 * initial_soc_for_interval

        initial_conc_neg_mol_m3 = initial_x_neg * self.c_n_max
        initial_conc_pos_mol_m3 = initial_y_pos_stoich * self.c_p_max

        min_conc_pos_mol_m3 = self.y_100 * self.c_p_max
        max_conc_pos_mol_m3 = self.y_0 * self.c_p_max

        initial_conc_pos_mol_m3 = np.clip(initial_conc_pos_mol_m3, min_conc_pos_mol_m3, max_conc_pos_mol_m3)

        current_parameters = self.parameters.copy()
        current_parameters.update({
            "Initial concentration in negative electrode [mol.m-3]": initial_conc_neg_mol_m3,
            "Initial concentration in positive electrode [mol.m-3]": initial_conc_pos_mol_m3,
        })

        sim = pybamm.Simulation(self.model, experiment=experiment, parameter_values=current_parameters,
                                  var_pts=self.var_pts, solver=pybamm.CasadiSolver(),
                                  output_variables=self.output_vars)

        try:
            solution = sim.solve()
        except pybamm.SolverError as e:
            print(f"Solver error during interval with power {power_kW} kW, initial SOC {initial_soc_for_interval}: {e}")
            return initial_soc_for_interval, V_pack_for_current_calc, 0.0, 0.0, np.array([0]), 0.0
        except Exception as e:
            print(f"An unexpected error occurred during interval with power {power_kW} kW, initial SOC {initial_soc_for_interval}: {e}")
            return initial_soc_for_interval, V_pack_for_current_calc, 0.0, 0.0, np.array([0]), 0.0

        x = solution["Average negative particle concentration"].entries
        soc = 100 * (x - self.x_0) / ((self.x_100 - self.x_0))

        voltage_cell = solution["Terminal voltage [V]"].entries
        current_cell = solution["Current [A]"].entries
        power_cell = solution["Power [W]"].entries
        time_points = solution["Time [s]"].entries
        neg_conc_mol_m3 = solution["Average negative particle concentration"].entries

        final_soc = soc[-1]
        final_voltage_cell = voltage_cell[-1]
        final_current_cell = current_cell[-1]
        final_power_cell = power_cell[-1]
        final_neg_conc = neg_conc_mol_m3[-1]

        final_voltage_pack = final_voltage_cell * self.Ns
        final_current_pack = final_current_cell * self.Np
        final_power_pack = final_power_cell * self.Ns * self.Np

        return final_soc, final_voltage_pack, final_current_pack, final_power_pack, time_points, final_neg_conc

def plot_simulation_results(simulation_output: Dict, price_profile: List[float], date_label: str = "Simulated Day", title_prefix: str = "Simulation"):
    """
    Plots the energy flows, battery SOC, and electricity prices from the simulation.
    Combines various energy flow types into a single subplot.
    """
    if "error" in simulation_output:
        print(f"Cannot plot: {simulation_output['error']}")
        return

    n_intervals_expected = 96
    interval_hours = simulation_output.get("interval_hours", 0.25)

    inverter_efficiency = simulation_output.get("inverter_efficiency", 0.95)
    pv_forecast_dc = simulation_output.get("original_pv_forecast_dc", [0.0] * n_intervals_expected)[:n_intervals_expected]
    pv_forecast_ac = [p * inverter_efficiency for p in pv_forecast_dc]

    time_intervals_flows = np.arange(0, n_intervals_expected * interval_hours, interval_hours)
    time_intervals_soc = np.arange(0, (n_intervals_expected + 1) * interval_hours, interval_hours)

    demand_profile = simulation_output.get("original_demand_profile_list", [0.0] * n_intervals_expected)[:n_intervals_expected]

    pv_to_load_ac = simulation_output.get("pv_to_load_ac", [0.0] * n_intervals_expected)[:n_intervals_expected]
    battery_to_load_ac = simulation_output.get("battery_to_load_ac", [0.0] * n_intervals_expected)[:n_intervals_expected]
    grid_to_load_ac = simulation_output.get("grid_to_load_ac", [0.0] * n_intervals_expected)[:n_intervals_expected]
    pv_to_battery_dc = simulation_output.get("pv_to_battery_dc", [0.0] * n_intervals_expected)[:n_intervals_expected]
    grid_to_battery_ac = simulation_output.get("grid_to_battery_ac", [0.0] * n_intervals_expected)[:n_intervals_expected]
    pv_to_grid_ac = simulation_output.get("pv_to_grid_ac", [0.0] * n_intervals_expected)[:n_intervals_expected]
    battery_to_grid_ac = simulation_output.get("battery_to_grid_ac", [0.0] * n_intervals_expected)[:n_intervals_expected]
    grid_import_ac = simulation_output.get("grid_import_ac", [0.0] * n_intervals_expected)[:n_intervals_expected]
    grid_export_ac = simulation_output.get("grid_export_ac", [0.0] * n_intervals_expected)[:n_intervals_expected]

    soc_data_plot = simulation_output.get("battery_soc_kwh_pybamm", simulation_output.get("battery_soc_kwh", []))
    if len(soc_data_plot) < (n_intervals_expected + 1):
        soc_data_plot.extend([soc_data_plot[-1] if soc_data_plot else 0.0] * ((n_intervals_expected + 1) - len(soc_data_plot)))
    soc_data_plot = soc_data_plot[:(n_intervals_expected + 1)]

    battery_capacity_kwh = simulation_output.get("battery_capacity_kwh", 10.24)
    battery_min_soc_kwh = simulation_output.get("battery_min_soc_kwh", 0.2 * battery_capacity_kwh)
    battery_max_soc_kwh = simulation_output.get("battery_max_soc_kwh", 0.8 * battery_capacity_kwh)

    fig, axs = plt.subplots(3, 1, figsize=(15, 15))

    axs[0].plot(time_intervals_flows, pv_forecast_ac, label='Total PV Generation (kW_ac)', color='orange', linestyle='--')
    axs[0].plot(time_intervals_flows, demand_profile, label='Total Appliance Demand (kW_ac)', color='red', linestyle=':')
    axs[0].plot(time_intervals_flows, pv_to_load_ac, label='PV to Load (kW_ac)', color='darkgreen', alpha=0.8)
    axs[0].plot(time_intervals_flows, battery_to_load_ac, label='Battery to Load (kW_ac)', color='purple', alpha=0.8)
    axs[0].plot(time_intervals_flows, grid_to_load_ac, label='Grid to Load (kW_ac)', color='blue', alpha=0.8)
    axs[0].plot(time_intervals_flows, [p * inverter_efficiency for p in pv_to_battery_dc], label='PV to Battery (kW_ac Eq.)', color='lightgreen', linestyle='-', alpha=0.7)
    axs[0].plot(time_intervals_flows, grid_to_battery_ac, label='Grid to Battery (kW_ac)', color='brown', linestyle='-', alpha=0.7)
    axs[0].plot(time_intervals_flows, pv_to_grid_ac, label='PV to Grid (kW_ac)', color='cyan', linestyle='--', alpha=0.7)
    axs[0].plot(time_intervals_flows, battery_to_grid_ac, label='Battery to Grid (kW_ac)', color='magenta', linestyle='--', alpha=0.7)
    axs[0].set_title(f'{title_prefix}: Combined Energy Flows for {date_label}')
    axs[0].set_xlabel('Time of Day (Hours)')
    axs[0].set_ylabel('Power (kW)')
    axs[0].legend(loc='upper right', ncol=2, fontsize='small')
    axs[0].grid(True, linestyle='--', alpha=0.8)
    axs[0].set_xticks(np.arange(0, (n_intervals_expected * interval_hours) + 1, 2))
    axs[0].set_xlim(0, n_intervals_expected * interval_hours)
    axs[0].set_ylim(bottom=0)

    axs[1].plot(time_intervals_soc, soc_data_plot, label='Battery SOC (kWh)', color='darkgreen', linewidth=2)
    axs[1].axhline(y=battery_capacity_kwh, color='grey', linestyle='--', label='Battery Max Capacity (kWh)')
    axs[1].axhline(y=battery_min_soc_kwh, color='red', linestyle='--', label='Battery Min SOC (kWh)')
    axs[1].axhline(y=battery_max_soc_kwh, color='green', linestyle='--', label='Battery Max Operable SOC (kWh)')
    axs[1].set_title(f'{title_prefix}: Battery State of Charge for {date_label}')
    axs[1].set_xlabel('Time of Day (Hours)')
    axs[1].set_ylabel('Battery SOC (kWh)')
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.8)
    axs[1].set_xticks(np.arange(0, (n_intervals_expected * interval_hours) + 1, 2))
    axs[1].set_xlim(0, n_intervals_expected * interval_hours)
    axs[1].set_ylim(0, battery_capacity_kwh * 1.1)

    if price_profile is not None and len(price_profile) >= n_intervals_expected:
        axs[2].plot(time_intervals_flows, price_profile[:n_intervals_expected], label='Electricity Price (Rs/kWh)', color='brown', linewidth=2)
        axs[2].set_title(f'Electricity Price Profile for {date_label}')
        axs[2].set_xlabel('Time of Day (Hours)')
        axs[2].set_ylabel('Price (Rs/kWh)')
        axs[2].legend()
        axs[2].grid(True, linestyle='--', alpha=0.8)
        axs[2].set_xticks(np.arange(0, (n_intervals_expected * interval_hours) + 1, 2))
        axs[2].set_xlim(0, n_intervals_expected * interval_hours)
    else:
        axs[2].set_title('Electricity Price Profile (Data Unavailable or Incomplete)')
        axs[2].set_xlabel('Time of Day (Hours)')
        axs[2].set_ylabel('Price (Rs/kWh)')
        axs[2].grid(True, linestyle='--', alpha=0.8)
        axs[2].set_xticks(np.arange(0, (n_intervals_expected * interval_hours) + 1, 2))
        axs[2].set_xlim(0, n_intervals_expected * interval_hours)

    plt.tight_layout()
    plt.show()


def plot_negative_electrode_concentration(
    time_points: List[float],
    neg_conc_mol_m3: List[float],
    date_label: str,
    title_prefix: str,
):
    """
    Plots the average negative electrode concentration over time.
    """
    if len(time_points) != len(neg_conc_mol_m3):
        print(f"Error plotting negative concentration: Length of time points ({len(time_points)}) does not match length of concentration data ({len(neg_conc_mol_m3)}). Cannot plot.")
        return


    plt.figure(figsize=(12, 6))
    plt.plot(time_points, neg_conc_mol_m3, label='Average Negative Particle Concentration (mol/m^3)', color='darkblue', linewidth=2)
    plt.title(f'{title_prefix}: Average Negative Electrode Concentration for {date_label}')
    plt.xlabel('Time (s)')
    plt.ylabel('Concentration (mol/m^3)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.8)

    max_time_sec = time_points[-1] if len(time_points) > 0 else 0
    plt.xticks(np.arange(0, max_time_sec + 1, 3600 * 2), labels=[f'{int(h)}:00' for h in np.arange(0, max_time_sec/3600 + 1, 2)])
    plt.xlim(0, max_time_sec)
    plt.tight_layout()
    plt.show()

# --- NEW: Simple Plotting Functions for Individual Data Streams ---
def plot_solar_forecast(pv_forecast_ac: List[float], date_label: str, interval_hours: float = 0.25):
    """
    Plots only the AC PV generation forecast.
    """
    n_intervals = len(pv_forecast_ac)
    time_intervals = np.arange(0, n_intervals * interval_hours, interval_hours)

    plt.figure(figsize=(12, 6))
    plt.plot(time_intervals, pv_forecast_ac, label='PV Generation (kW_ac)', color='orange', linewidth=2)
    plt.title(f'Solar PV Generation Forecast for {date_label}')
    plt.xlabel('Time of Day (Hours)')
    plt.ylabel('Power (kW)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.8)
    plt.xticks(np.arange(0, (n_intervals * interval_hours) + 1, 2))
    plt.xlim(0, n_intervals * interval_hours)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.show()

def plot_single_price_profile(price_profile: List[float], date_label: str, market_type: str, interval_hours: float = 0.25):
    """
    Plots only the electricity price profile.
    """
    n_intervals = len(price_profile)
    time_intervals = np.arange(0, n_intervals * interval_hours, interval_hours)

    plt.figure(figsize=(12, 6))
    plt.plot(time_intervals, price_profile, label=f'{market_type.capitalize()} Price (Rs/kWh)', color='brown', linewidth=2)
    plt.title(f'Electricity Price Profile for {date_label} ({market_type.capitalize()} Market)')
    plt.xlabel('Time of Day (Hours)')
    plt.ylabel('Price (Rs/kWh)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.8)
    plt.xticks(np.arange(0, (n_intervals * interval_hours) + 1, 2))
    plt.xlim(0, n_intervals * interval_hours)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.show()

# --- 1. Solar PV Generation Forecasting Function ---
def generate_pv_forecast(
    solar_excel: str,
    selected_day: str,
    lat: float = 17.4065,
    lon: float = 78.4772,
    alt: float = 536,
    tz: str = 'Asia/Kolkata',
    surface_tilt: float = 30,
    surface_azimuth: float = 180,
    module_name: str = 'Silevo_Triex_U300_Black__2014_',
    inverter_name: str = 'ABB__PVI_3_0_OUTD_S_US_A__208V_',
    strings_per_inverter: int = 4,
    modules_per_string: int = 3
) -> dict:
    """
    Calculates the DC power generation forecast for a specified day using pvlib.
    """
    print(f"DEBUG: generate_pv_forecast called with solar_excel='{solar_excel}' for selected_day='{selected_day}'")
    if not os.path.exists(solar_excel):
        print(f"ERROR: Solar data file not found at: {solar_excel}")
        return {"error": f"Solar data file not found at: {solar_excel}. Please ensure it's accessible."}

    try:
        solar_data = pd.read_excel(solar_excel)
        solar_data['Datetime'] = pd.to_datetime(solar_data['Datetime_solar'], format='%d-%m-%Y %H:%M:%S')
        solar_data.set_index('Datetime', inplace=True)
        print(f"DEBUG: Successfully loaded solar data from {solar_excel}. Shape: {solar_data.shape}")

        if solar_data.index.tz is None:
            try:
                import pytz
                kolkata_tz = pytz.timezone(tz)
                solar_data.index = solar_data.index.tz_localize(kolkata_tz)
            except Exception as e:
                return {"error": f"Failed to localize timezone for solar data: {e}. Ensure 'pytz' is installed and timezone '{tz}' is valid."}

    except KeyError as e:
        return {"error": f"Missing expected column in solar data file: {e}. Ensure 'Datetime_solar', 'ghi', 'dni', 'dhi', 'air_temp', 'wind speed 10m' columns exist."}
    except Exception as e:
        return {"error": f"Error loading or processing solar data from '{solar_excel}': {e}"}

    try:
        daily_data = solar_data.loc[selected_day]
        if daily_data.empty:
            print(f"ERROR: No data found for the selected day: {selected_day} in '{solar_excel}'.")
            return {"error": f"No data found for the selected day: {selected_day} in '{solar_excel}'."}
        print(f"DEBUG: Found daily solar data for {selected_day}. Shape: {daily_data.shape}")
    except KeyError:
        print(f"ERROR: No data found for the selected day: {selected_day}. Please ensure the date format is ISO (YYYY-MM-DD) and it exists in the data loaded from '{solar_excel}'.")
        return {"error": f"No data found for the selected day: {selected_day}. Please ensure the date format is ISO (YYYY-MM-DD) and it exists in the data loaded from '{solar_excel}'."}
    except Exception as e:
        print(f"ERROR: Error filtering data for selected day: {e}. Check the date format or data availability.")
        return {"error": f"Error filtering data for selected day: {e}. Check the date format or data availability."}

    solpos = pvlib.solarposition.get_solarposition(daily_data.index, lat, lon, alt)

    if solpos['apparent_zenith'].isnull().all() or solpos['elevation'].isnull().all():
        print(f"Warning: Solar position data is all NaN for {selected_day}. This might indicate the sun is below the horizon for the entire day.")
        return {"dc_power_forecast": [0.0] * 96}

    weather = pd.DataFrame({
        'ghi': daily_data['ghi'], 'dni': daily_data['dni'], 'dhi': daily_data['dhi'],
        'temp_air': daily_data['air_temp'],
        'wind_speed': daily_data['wind speed 10m'],
        'apparent_zenith': solpos['apparent_zenith'], 'elevation': solpos['elevation']
    }, index=daily_data.index)


    try:
        mods = pvlib.pvsystem.retrieve_sam('sandiamod')
        invs = pvlib.pvsystem.retrieve_sam('cecinverter')
        module_params = mods[module_name]
        inverter_params = invs[inverter_name]
        tparams = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    except KeyError as e:
        return {"error": f"Invalid module ('{module_name}') or inverter ('{inverter_name}') name provided: {e}. Please check names from pvlib.pvsystem.retrieve_sam()."}
    except Exception as e:
        return {"error": f"Error retrieving SAM parameters: {e}"}

    system = pvlib.pvsystem.PVSystem(
        surface_tilt=surface_tilt, surface_azimuth=surface_azimuth,
        module_parameters=module_params,
        inverter_parameters=inverter_params,
        temperature_model_parameters = tparams,
        strings_per_inverter=strings_per_inverter,
        modules_per_string=modules_per_string
    )

    location = pvlib.location.Location(lat, lon, tz=tz, altitude=alt)

    mc = pvlib.modelchain.ModelChain.with_sapm(system, location)
    try:
        mc.run_model(weather)
    except Exception as e:
        return {"error": f"Error running pvlib model: {e}. This might be due to issues with weather data or system parameters."}

    try:
        start_of_day = pd.to_datetime(selected_day).normalize()
        import pytz
        kolkata_tz = pytz.timezone(tz)
        start_of_day = start_of_day.tz_localize(kolkata_tz)
    except ValueError as e:
        return {"error": f"Error parsing or localizing selected day '{selected_day}': {e}. Ensure it's in ISO-MM-DD format and '{tz}' is a valid timezone string."}
    except Exception as e:
        return {"error": f"Error during timezone localization in generate_pv_forecast: {e}. Ensure pytz is installed."}


    full_day_index = pd.date_range(start=start_of_day, periods=96, freq='15min', tz=tz)

    dc_power_series = pd.Series(0.0, index=full_day_index)

    if 'p_mp' in mc.results.dc:
        calculated_dc_power = mc.results.dc['p_mp'].resample('15min').mean().fillna(0) / 1000.0
        if calculated_dc_power.index.tz is None:
            try:
                import pytz
                kolkata_tz = pytz.timezone(tz)
                calculated_dc_power.index = calculated_dc_power.index.tz_localize(kolkata_tz)
            except Exception as e:
                print(f"Warning: Could not localize calculated_dc_power index: {e}")
                pass
        dc_power_series.update(calculated_dc_power)
    else:
        return {"error": "Expected 'p_mp' in pvlib results but it was not found."}

    dc_power_forecast_list = dc_power_series.tolist()

    return {
        "dc_power_forecast": dc_power_forecast_list,
        "original_pv_forecast_dc": dc_power_forecast_list,
        "interval_hours": 0.25
    }

# New Helper Function: Non-recursive demand profile calculation ---
def _calculate_single_demand_profile(appliances_data: list, n_slots: int = 96, inverter_efficiency: float = 0.95, pv_forecast_list: List[float] = None, user_preferences: Dict = None, current_date: datetime.date = None) -> Tuple[List[float], Dict[str, List[float]]]:
    """
    Helper function to calculate a single daily electricity demand profile based on
    appliance schedules and power consumption, considering PV-aware shifting.
    """
    profile = np.zeros(n_slots)
    individual_appliance_power_per_slot = {
        appliance["name"]: np.zeros(n_slots).tolist() for appliance in appliances_data
    }

    if current_date:
        is_weekend = current_date.weekday() >= 5
    else:
        is_weekend = False

    for appliance in appliances_data:
        if is_weekend:
            appliance_schedules_for_day = appliance.get("schedule_weekend", appliance.get("schedule_weekday", []))
        else:
            appliance_schedules_for_day = appliance.get("schedule_weekday", appliance.get("schedule_weekend", []))

        for schedule in appliance_schedules_for_day:
            start_slot = int(schedule["start"] * 4)
            end_slot = int(schedule["end"] * 4)
            raw_duration_slots = int(appliance["duration"] * 4)
            appliance_power_ac = appliance["power"]

            window_size = end_slot - start_slot
            effective_duration_slots = min(raw_duration_slots, max(0, window_size))

            if effective_duration_slots <= 0 or window_size <= 0:
                continue

            max_possible_start_slot = end_slot - effective_duration_slots

            start_time_slot = -1

            if user_preferences and appliance["name"] in user_preferences.get("preferred_appliance_times", {}):
                preferred_hour = user_preferences["preferred_appliance_times"][appliance["name"]]
                preferred_slot = int(preferred_hour * 4)
                if start_slot <= preferred_slot <= end_slot - effective_duration_slots:
                    if pv_forecast_list is not None:
                        is_pv_available_at_preferred_time = True
                        for t in range(preferred_slot, preferred_slot + effective_duration_slots):
                            if t >= n_slots or (pv_forecast_list[t] * inverter_efficiency) < appliance_power_ac:
                                is_pv_available_at_preferred_time = False
                                break
                        if is_pv_available_at_preferred_time:
                            start_time_slot = preferred_slot
                    else:
                        start_time_slot = preferred_slot

            if start_time_slot == -1 and appliance["name"] in ["Water Pump (1hp)", "Washing Machine", "Dish Washer", "Geyser (10 l)", "Electrical Cooker"] and pv_forecast_list is not None:
                best_pv_start_slot = -1
                max_pv_power_in_window = -1.0

                for candidate_start_slot in range(start_slot, max_possible_start_slot + 1):
                    if candidate_start_slot + effective_duration_slots > n_slots:
                        continue

                    current_pv_power_sum = 0.0
                    for t in range(candidate_start_slot, candidate_start_slot + effective_duration_slots):
                        current_pv_power_sum += pv_forecast_list[t] * inverter_efficiency

                    if current_pv_power_sum > max_pv_power_in_window:
                        max_pv_power_in_window = current_pv_power_sum
                        best_pv_start_slot = candidate_start_slot

                if best_pv_start_slot != -1:
                    start_time_slot = best_pv_start_slot

            if start_time_slot == -1:
                if max_possible_start_slot < start_slot:
                     start_time_slot = start_slot
                else:
                    start_time_slot = np.random.randint(start_slot, max_possible_start_slot + 1)

            for t in range(start_time_slot, min(start_time_slot + effective_duration_slots, n_slots)):
                if 0 <= t < n_slots:
                    profile[t] += appliance_power_ac / 1000
                    individual_appliance_power_per_slot[appliance['name']][t] = appliance_power_ac / 1000

    return profile.tolist(), individual_appliance_power_per_slot


# --- 2. Appliance Demand Profile Generation Function (now uses helper) ---
def generate_appliance_demand_profiles(n_days: int = 1, n_slots: int = 96, appliances_data: list = None, pv_forecast_list: List[float] = None, inverter_efficiency: float = 0.95, user_preferences: Dict = None, current_date: datetime.date = None):
    """
    Generates a specified number of daily electricity demand profiles based on
    appliance schedules and power consumption. Now correctly handles weekday/weekend.
    """
    if appliances_data == None:
        appliances_data = DEFAULT_APPLIANCES_DATA

    demand_profiles = []
    last_individual_appliance_power = {appliance["name"]: np.zeros(n_slots).tolist() for appliance in appliances_data}

    if pv_forecast_list is not None and len(pv_forecast_list) != n_slots:
        print(f"Warning: pv_forecast_list length ({len(pv_forecast_list)}) does not match n_slots ({n_slots}). PV-aware scheduling may not work as expected.")
        pv_forecast_list = None

    for day_offset in range(n_days):
        current_day = current_date + datetime.timedelta(days=day_offset) if current_date else datetime.date.today()

        profile, individual_power = _calculate_single_demand_profile(
            appliances_data=appliances_data,
            n_slots=n_slots,
            inverter_efficiency=inverter_efficiency,
            pv_forecast_list=pv_forecast_list,
            user_preferences=user_preferences,
            current_date=current_day
        )
        demand_profiles.append(profile)
        last_individual_appliance_power = individual_power

    demand_profiles_array = np.array(demand_profiles)
    avg_power_consumption = np.mean(demand_profiles_array, axis=0)

    return {
        "average_power_consumption": avg_power_consumption.tolist(),
        "individual_appliance_power_per_slot": last_individual_appliance_power
    }


# --- 3. IEX Market Clearing Price (MCP) Fetching Function ---
def get_iex_mcp(date_str: str = None, market_type: str = 'real-time') -> dict:
    """
    Fetches the Market Clearing Price (MCP) profile from IEX India for a given date.
    Returns available data even if incomplete, along with a 'data_complete' flag.
    """
    if date_str is None:
        date_str = datetime.datetime.now().strftime("%d-%m-%Y")

    base_url = "https://www.iexindia.com/market-data/"
    if market_type == 'real-time':
        market_segment_url = "real-time-market/market-snapshot"
    elif market_type == 'day-ahead':
        market_segment_url = "day-ahead-market/market-snapshot"
    else:
        return {"error": "Invalid market_type. Please use 'real-time' or 'day-ahead'."}

    url = f"{base_url}{market_segment_url}?interval=ONE_FOURTH_HOUR&dp=SELECT_RANGE&showGraph=false&toDate={date_str}&fromDate={date_str}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")
        table = soup.find("table")

        if not table:
            return {"error": f"No data table found on IEX page for {date_str} ({market_type} market). Website structure might have changed or data is unavailable."}

        rows = table.find_all("tr")[1:]
        data = []
        rowspan_tracker = {}

        for row in rows:
            cells = row.find_all("td")
            col_index = 0
            row_data = []

            temp_rowspan_tracker = dict(rowspan_tracker)
            for stored_col_index, stored_info in temp_rowspan_tracker.items():
                if stored_info["span"] > 0:
                    current_idx = 0
                    while current_idx < len(row_data) and current_idx < stored_col_index:
                        current_idx += 1
                    row_data.insert(current_idx, stored_info["value"])
                    rowspan_tracker[stored_col_index]["span"] -= 1
                else:
                    del rowspan_tracker[stored_col_index]

            for cell in cells:
                text_value = cell.text.strip()
                if cell.has_attr("rowspan"):
                    rowspan = int(cell["rowspan"])
                    actual_col_index = len(row_data)
                    rowspan_tracker[actual_col_index] = {"value": text_value, "span": rowspan - 1}
                    row_data.append(text_value)
                else:
                    row_data.append(text_value)

            price_found = False
            for val in reversed(row_data):
                try:
                    price_mwh = float(val.replace(",", "").strip())
                    data.append(price_mwh / 1000)
                    price_found = True
                    break
                except ValueError:
                    continue

            if not price_found:
                print(f"Warning: Could not find a parsable price value in row for {date_str} ({market_type} market). Skipping row: {row_data}")
                continue

        data_complete = (len(data) == 96)
        return {"price_profile": data, "data_complete": data_complete}

    except requests.exceptions.RequestException as e:
        return {"error": f"Network or HTTP error fetching IEX data for {date_str} ({market_type} market): {e}. The website might be down or your IP is blocked."}
    except Exception as e:
        return {"error": f"An unexpected error occurred while fetching IEX data for {date_str} ({market_type} market): {e}. Website structure may have changed."}

# --- 4. Simulate Solar Load Serving and Battery Interaction (REACTIVE, PRICE-BLIND BASELINE) ---
def baseline_reactive_price_blind(
    pv_forecast_list: List[float],
    demand_profile_list: List[float],
    params: Dict
) -> Dict:
    """
    Simulates energy flow for the baseline (reactive, price-blind) scenario.
    This version now accepts a 'params' dictionary for consistency.
    """
    # --- Extract parameters from the dictionary ---
    interval_hours = params["interval_hours"]
    inverter_efficiency = params["inverter_efficiency"]
    battery_capacity_kwh = params["battery_capacity_kwh"]
    battery_max_charge_rate_kw = params["battery_max_charge_rate_kw"]
    battery_max_discharge_rate_kw = params["battery_max_discharge_rate_kw"]
    battery_efficiency = params["battery_efficiency"]
    initial_soc = params["initial_soc"]
    battery_min_soc_kwh = params["battery_min_soc_kwh"]
    battery_max_soc_kwh = params["battery_max_soc_kwh"]

    n_slots = len(pv_forecast_list)

    local_battery_sim = BatterySimulator(
        Ns=params['Ns'],
        Np=params['Np'],
        V_cell_nominal=params['V_cell_nominal']
    )
    initial_x_neg_initial = local_battery_sim.x_0 + (local_battery_sim.x_100 - local_battery_sim.x_0) * (initial_soc / battery_capacity_kwh)

    current_soc_kwh = initial_soc
    current_soc_perc = (initial_soc / battery_capacity_kwh) * 100.0
    current_V_pack = params['V_cell_nominal'] * params['Ns']

    sim_pv_to_load_ac = []
    sim_pv_to_battery_dc = []
    sim_pv_to_grid_ac = []
    sim_battery_to_load_ac = []
    sim_battery_to_grid_ac = []
    sim_grid_to_load_ac = []
    sim_grid_to_battery_ac = []
    sim_grid_import_ac = []
    sim_grid_export_ac = []
    sim_battery_soc_kwh_tracker = [initial_soc]
    sim_battery_voltage_output = [current_V_pack]
    sim_battery_current_output = [0.0]
    sim_neg_conc_tracker = [initial_x_neg_initial]
    sim_time_tracker = [0]


    for i in range(n_slots):
        Pdc = pv_forecast_list[i]
        L = demand_profile_list[i]
        Pac_pv = Pdc * inverter_efficiency

        cur_pv_to_load = 0.0
        cur_pv_to_battery_dc = 0.0
        cur_pv_to_grid = 0.0
        cur_battery_to_load = 0.0
        cur_grid_to_load = 0.0
        cur_grid_import_total = 0.0
        cur_grid_export_total = 0.0

        # --- Energy Flow Prioritization ---
        # 1. Serve Load from PV
        load_met_by_pv = min(Pac_pv, L)
        cur_pv_to_load = load_met_by_pv
        remaining_load = L - load_met_by_pv
        pv_surplus_ac = Pac_pv - load_met_by_pv

        # 2. Charge Battery from PV Surplus (AC equivalent for charge rate calculation)
        charge_potential_ac = min(pv_surplus_ac, battery_max_charge_rate_kw)
        max_charge_capacity_kwh = battery_max_soc_kwh - current_soc_kwh
        charge_limited_by_capacity_ac = max_charge_capacity_kwh / interval_hours / battery_efficiency

        actual_charge_ac = min(charge_potential_ac, charge_limited_by_capacity_ac)
        cur_pv_to_battery_dc = actual_charge_ac / inverter_efficiency
        pv_surplus_after_charge_ac = pv_surplus_ac - actual_charge_ac

        pybamm_power_from_pv_charge_kw = -cur_pv_to_battery_dc

        # 3. Meet Remaining Load from Battery (Discharge)
        discharge_potential_ac = min(remaining_load, battery_max_discharge_rate_kw)
        max_discharge_capacity_kwh = current_soc_kwh - battery_min_soc_kwh
        discharge_limited_by_capacity_ac = max_discharge_capacity_kwh / interval_hours * battery_efficiency

        actual_discharge_ac = min(discharge_potential_ac, discharge_limited_by_capacity_ac)
        cur_battery_to_load = actual_discharge_ac
        remaining_load_after_battery = remaining_load - actual_discharge_ac

        pybamm_power_from_battery_discharge_kw = cur_battery_to_load / battery_efficiency

        net_battery_power_for_pybamm_kw = pybamm_power_from_battery_discharge_kw + pybamm_power_from_pv_charge_kw

        # 4. Handle remaining energy (Grid interaction)
        if remaining_load_after_battery > 0:
            cur_grid_to_load = remaining_load_after_battery
            cur_grid_import_total = cur_grid_to_load
            cur_grid_export_total = 0.0
        elif pv_surplus_after_charge_ac > 0:
            cur_pv_to_grid = pv_surplus_after_charge_ac
            cur_grid_export_total = cur_pv_to_grid
            cur_grid_import_total = 0.0
        else:
            cur_grid_to_load = 0.0
            cur_pv_to_grid = 0.0
            cur_grid_import_total = 0.0
            cur_grid_export_total = 0.0

        final_soc_perc, final_voltage_pack, final_current_pack, final_power_pack, time_points, final_neg_conc = \
            local_battery_sim.run_single_interval_simulation(
                net_battery_power_for_pybamm_kw,
                current_soc_perc / 100.0,
                current_V_pack
            )

        current_soc_perc = final_soc_perc
        current_soc_kwh = (current_soc_perc / 100.0) * battery_capacity_kwh
        current_V_pack = final_voltage_pack

        sim_pv_to_load_ac.append(round(float(cur_pv_to_load), 3))
        sim_pv_to_battery_dc.append(round(float(cur_pv_to_battery_dc), 3))
        sim_pv_to_grid_ac.append(round(float(cur_pv_to_grid), 3))
        sim_battery_to_load_ac.append(round(float(cur_battery_to_load), 3))
        sim_battery_to_grid_ac.append(0.0) # Price-blind baseline never sells battery to grid
        sim_grid_to_load_ac.append(round(float(cur_grid_to_load), 3))
        sim_grid_to_battery_ac.append(0.0) # Price-blind baseline never buys from grid to charge
        sim_grid_import_ac.append(round(float(cur_grid_import_total), 3))
        sim_grid_export_ac.append(round(float(cur_grid_export_total), 3))

        sim_battery_soc_kwh_tracker.append(round(float(current_soc_kwh), 3))
        sim_battery_voltage_output.append(round(float(current_V_pack), 3))
        sim_battery_current_output.append(round(float(final_current_pack), 3))
        sim_neg_conc_tracker.append(final_neg_conc)
        sim_time_tracker.append(sim_time_tracker[-1] + time_points[-1] if time_points.size > 0 else sim_time_tracker[-1])

    total_demand_kwh = sum(demand_profile_list) * interval_hours
    total_pv_gen_ac = sum([p * inverter_efficiency for p in pv_forecast_list]) * interval_hours

    energy_pv_to_load = sum(sim_pv_to_load_ac) * interval_hours
    energy_pv_to_battery_dc = sum(sim_pv_to_battery_dc) * interval_hours
    energy_pv_to_battery_ac_equiv = energy_pv_to_battery_dc * inverter_efficiency

    self_consumption = ((energy_pv_to_load + energy_pv_to_battery_ac_equiv) / total_pv_gen_ac) * 100 if total_pv_gen_ac > 0 else 0
    self_consumption = round(max(0, min(100, self_consumption)), 2)

    energy_from_pv_to_load_ac = sum(sim_pv_to_load_ac) * interval_hours
    energy_from_battery_to_load_ac = sum(sim_battery_to_load_ac) * interval_hours

    self_sufficiency = ((energy_from_pv_to_load_ac + energy_from_battery_to_load_ac) / total_demand_kwh) * 100 if total_demand_kwh > 0 else 0
    self_sufficiency = round(max(0, min(100, self_sufficiency)), 2)

    return {
        "pv_to_load_ac": sim_pv_to_load_ac,
        "pv_to_battery_dc": sim_pv_to_battery_dc,
        "pv_to_grid_ac": sim_pv_to_grid_ac,
        "battery_to_load_ac": sim_battery_to_load_ac,
        "battery_to_grid_ac": sim_battery_to_grid_ac,
        "grid_to_load_ac": sim_grid_to_load_ac,
        "grid_to_battery_ac": sim_grid_to_battery_ac,
        "grid_import_ac": sim_grid_import_ac,
        "grid_export_ac": sim_grid_export_ac,
        "battery_soc_kwh": sim_battery_soc_kwh_tracker,
        "battery_voltage": sim_battery_voltage_output,
        "battery_current": sim_battery_current_output,
        "interval_hours": interval_hours,
        "inverter_efficiency": inverter_efficiency,
        "battery_capacity_kwh": battery_capacity_kwh,
        "battery_max_charge_rate_kw": battery_max_charge_rate_kw,
        "battery_max_discharge_rate_kw": battery_max_discharge_rate_kw,
        "battery_efficiency": battery_efficiency,
        "initial_soc": initial_soc,
        "demand_profile_list": demand_profile_list, # Renamed for clarity
        "original_demand_profile_list": demand_profile_list,
        "original_pv_forecast_dc": pv_forecast_list,
        "battery_min_soc_kwh": battery_min_soc_kwh,
        "battery_max_soc_kwh": battery_max_soc_kwh,
        "neg_particle_concentration_mol_m3": sim_neg_conc_tracker,
        "time_points_sec": sim_time_tracker,
        "battery_soc_kwh_pybamm": sim_battery_soc_kwh_tracker,
        "battery_voltage_pybamm": sim_battery_voltage_output,
        "battery_current_pybamm": sim_battery_current_output,
        # "battery_power_pybamm": net_battery_power_for_pybamm_kw, # This is only the last value, might be confusing
        "self_consumption (%)": self_consumption,
        "self_sufficiency (%)": self_sufficiency
    }


# --- 5. NEW: HEURISTIC (PRICE-AWARE) RULE-BASED BASELINE ---
def heuristic_price_aware_baseline(
    pv_forecast_list: List[float],
    demand_profile_list: List[float],
    price_profile_list: List[float],
    params: Dict
) -> Dict:
    """
    Simulates energy flow for a heuristic (rule-based, price-aware) scenario.
    This is the "stronger baseline" requested by the reviewer.
    """
    # --- Extract parameters from the dictionary ---
    interval_hours = params["interval_hours"]
    inverter_efficiency = params["inverter_efficiency"]
    battery_capacity_kwh = params["battery_capacity_kwh"]
    battery_max_charge_rate_kw = params["battery_max_charge_rate_kw"]
    battery_max_discharge_rate_kw = params["battery_max_discharge_rate_kw"]
    battery_efficiency = params["battery_efficiency"]
    initial_soc = params["initial_soc"]
    battery_min_soc_kwh = params["battery_min_soc_kwh"]
    battery_max_soc_kwh = params["battery_max_soc_kwh"]

    n_slots = len(pv_forecast_list)

    # --- NEW: Define Price Thresholds ---
    price_array = np.array(price_profile_list)
    LOW_PRICE_THRESHOLD = np.percentile(price_array, 25)  # e.g., prices in the bottom 25%
    HIGH_PRICE_THRESHOLD = np.percentile(price_array, 75) # e.g., prices in the top 75%
    print(f"DEBUG (Heuristic): Low Price Threshold: {LOW_PRICE_THRESHOLD:.2f} Rs/kWh, High Price Threshold: {HIGH_PRICE_THRESHOLD:.2f} Rs/kWh")

    local_battery_sim = BatterySimulator(
        Ns=params['Ns'],
        Np=params['Np'],
        V_cell_nominal=params['V_cell_nominal']
    )
    initial_x_neg_initial = local_battery_sim.x_0 + (local_battery_sim.x_100 - local_battery_sim.x_0) * (initial_soc / battery_capacity_kwh)

    current_soc_kwh = initial_soc
    current_soc_perc = (initial_soc / battery_capacity_kwh) * 100.0
    current_V_pack = params['V_cell_nominal'] * params['Ns']

    sim_pv_to_load_ac = []
    sim_pv_to_battery_dc = []
    sim_pv_to_grid_ac = []
    sim_battery_to_load_ac = []
    sim_battery_to_grid_ac = []
    sim_grid_to_load_ac = []
    sim_grid_to_battery_ac = []
    sim_grid_import_ac = []
    sim_grid_export_ac = []
    sim_battery_soc_kwh_tracker = [initial_soc]
    sim_battery_voltage_output = [current_V_pack]
    sim_battery_current_output = [0.0]
    sim_neg_conc_tracker = [initial_x_neg_initial]
    sim_time_tracker = [0]


    for i in range(n_slots):
        Pdc = pv_forecast_list[i]
        L = demand_profile_list[i]
        Pac_pv = Pdc * inverter_efficiency
        current_price = price_profile_list[i] # Get current price

        # --- MODIFIED LOGIC START ---

        # --- Efficiency setup (adjust if you already have seperate values) ---
        η_inv = inverter_efficiency             # inverter: AC <-> DC
        η_round = battery_efficiency            # round-trip efficiency (0..1); if you already have η_ch/η_dis, use them
        η_ch = η_dis = float(np.sqrt(η_round))      # split round-trip equally (common default)

        # --- Battery Headroom Calculations (Helper logic) ---
        # Energetic headroom in kWh
        charge_headroom_kwh = max(0.0, battery_max_soc_kwh - current_soc_kwh)
        discharge_headroom_kwh = max(0.0, current_soc_kwh - battery_min_soc_kwh)

        # Convert the energy headroom to *AC-side* power headroom (kW) for this interval:
        #   AC -> inverter -> DC -> battery (charging)
        #   stored_energy_added (kWh) = P_ac_charge * η_inv * η_ch * interval_hours
        # => P_ac_charge_max_from_soc = charge_headroom_kwh / (η_inv * η_ch * interval_hours)
        if interval_hours > 0:
            max_charge_power_from_soc_ac = charge_headroom_kwh / (η_inv * η_ch * interval_hours)
            max_discharge_power_from_soc_ac = (discharge_headroom_kwh / interval_hours) * (η_inv * η_dis)
        else:
            max_charge_power_from_soc_ac = 0.0
            max_discharge_power_from_soc_ac = 0.0

        # Enforce inverter/battery power limits (AC-side limits)
        actual_charge_headroom_ac = max(0.0, min(battery_max_charge_rate_kw, max_charge_power_from_soc_ac))
        actual_discharge_headroom_ac = max(0.0, min(battery_max_discharge_rate_kw, max_discharge_power_from_soc_ac))

        # --- Energy Flow Prioritization (Step 1) ---
        # Use available PV (AC-side) to serve load first
        load_met_by_pv = min(Pac_pv, L)
        cur_pv_to_load = load_met_by_pv
        remaining_load = L - load_met_by_pv
        pv_surplus_ac = max(Pac_pv - load_met_by_pv, 0.0)

        # init flows (AC-side)
        cur_pv_to_battery_ac = 0.0
        cur_grid_to_battery_ac = 0.0
        cur_battery_to_load_ac = 0.0
        cur_battery_to_grid_ac = 0.0
        cur_pv_to_grid = 0.0
        cur_grid_to_load = 0.0

        # net_battery_power_for_pybamm_kw: positive => DC removed from battery (discharge),
        # negative => DC added to battery (charge)
        net_battery_power_for_pybamm_kw = 0.0

        # --- Energy Flow Prioritization (Step 2: Price-Based Logic) ---
        if current_price <= LOW_PRICE_THRESHOLD:
            # --- MODE: BUY/CHARGE (Price is Low) ---
            # 2a. Charge from PV surplus (AC)
            charge_from_pv_ac = min(pv_surplus_ac, actual_charge_headroom_ac)
            cur_pv_to_battery_ac = charge_from_pv_ac
            pv_surplus_ac -= charge_from_pv_ac
            actual_charge_headroom_ac -= charge_from_pv_ac

            # 2b. Charge from Grid (AC) if still room
            if actual_charge_headroom_ac > 0.0:
                cur_grid_to_battery_ac = actual_charge_headroom_ac
                actual_charge_headroom_ac = 0.0

            # 2c. Meet Remaining Load from Grid (do NOT discharge battery in BUY mode)
            cur_grid_to_load = remaining_load
            remaining_load = 0.0

        elif current_price >= HIGH_PRICE_THRESHOLD:
            # --- MODE: SELL/DISCHARGE (Price is High) ---
            # 2a. Use battery to meet remaining load first (AC)
            discharge_to_load_ac = min(remaining_load, actual_discharge_headroom_ac)
            cur_battery_to_load_ac = discharge_to_load_ac
            remaining_load -= discharge_to_load_ac
            actual_discharge_headroom_ac -= discharge_to_load_ac

            # 2b. If still load remaining, buy from grid
            if remaining_load > 0.0:
                cur_grid_to_load = remaining_load
                remaining_load = 0.0

            # 2c. Sell PV surplus to grid (AC)
            cur_pv_to_grid = pv_surplus_ac
            pv_surplus_ac = 0.0

            # 2d. Sell remaining battery (AC) to grid (limited by remaining discharge headroom)
            if actual_discharge_headroom_ac > 0.0:
                cur_battery_to_grid_ac = actual_discharge_headroom_ac
                actual_discharge_headroom_ac = 0.0

        else:
            # --- MODE: SELF-CONSUME (Price is Medium) ---
            # 2a. Charge from PV Surplus (AC)
            charge_from_pv_ac = min(pv_surplus_ac, actual_charge_headroom_ac)
            cur_pv_to_battery_ac = charge_from_pv_ac
            pv_surplus_ac -= charge_from_pv_ac
            actual_charge_headroom_ac -= charge_from_pv_ac

            # 2b. Meet Remaining Load from Battery (AC)
            discharge_to_load_ac = min(remaining_load, actual_discharge_headroom_ac)
            cur_battery_to_load_ac = discharge_to_load_ac
            remaining_load -= discharge_to_load_ac
            actual_discharge_headroom_ac -= discharge_to_load_ac

            # 2c. Meet Final Remaining Load from Grid (AC)
            if remaining_load > 0.0:
                cur_grid_to_load = remaining_load
                remaining_load = 0.0

            # 2d. Send any final PV surplus to grid
            cur_pv_to_grid = pv_surplus_ac
            pv_surplus_ac = 0.0

        # --- Safety: Prevent simultaneous battery charge & discharge in same interval ---
        total_charge_ac = cur_pv_to_battery_ac + cur_grid_to_battery_ac
        total_discharge_ac = cur_battery_to_load_ac + cur_battery_to_grid_ac
        if total_charge_ac > 0.0 and total_discharge_ac > 0.0:
            # Cancel the smaller flow (prefer meeting load if there is discharge)
            if total_discharge_ac >= total_charge_ac:
                # Cancel charging flows
                cur_pv_to_battery_ac = 0.0
                cur_grid_to_battery_ac = 0.0
            else:
                # Cancel discharging flows
                cur_battery_to_load_ac = 0.0
                cur_battery_to_grid_ac = 0.0

        # --- Convert AC flows <-> DC and compute net battery DC power for PyBaMM ---
        # DC energy stored per AC charge power (kW) = P_ac * η_inv * η_ch  (kW DC stored rate)
        dc_charge_from_pv_kw = cur_pv_to_battery_ac * η_inv * η_ch
        dc_charge_from_grid_kw = cur_grid_to_battery_ac * η_inv * η_ch
        DC_in_stored_kw = dc_charge_from_pv_kw + dc_charge_from_grid_kw

        # DC removed from battery per AC discharge power (kW) = P_ac / (η_inv * η_dis)
        dc_removed_for_load_kw = (cur_battery_to_load_ac) / (η_inv * η_dis) if η_inv * η_dis > 0 else 0.0
        dc_removed_for_grid_kw = (cur_battery_to_grid_ac) / (η_inv * η_dis) if η_inv * η_dis > 0 else 0.0
        DC_out_removed_kw = dc_removed_for_load_kw + dc_removed_for_grid_kw

        # net: positive => discharge (DC removed), negative => charge (DC added)
        net_battery_power_for_pybamm_kw = DC_out_removed_kw - DC_in_stored_kw

        # --- Grid Import/Export Calculation (AC-side totals) ---
        cur_grid_import_total = cur_grid_to_load + cur_grid_to_battery_ac      # AC imported this interval
        cur_grid_export_total = cur_pv_to_grid + cur_battery_to_grid_ac      # AC exported this interval

        # --- MODIFIED LOGIC END ---

        # --- PyBaMM Simulation Call ---
        final_soc_perc, final_voltage_pack, final_current_pack, final_power_pack, time_points, final_neg_conc = \
            local_battery_sim.run_single_interval_simulation(
                net_battery_power_for_pybamm_kw,
                current_soc_perc / 100.0,
                current_V_pack
            )

        current_soc_perc = final_soc_perc
        current_soc_kwh = (current_soc_perc / 100.0) * battery_capacity_kwh
        # Add a safety clip for SOC in case of simulation overshoot
        current_soc_kwh = max(battery_min_soc_kwh, min(battery_max_soc_kwh, current_soc_kwh))
        current_soc_perc = (current_soc_kwh / battery_capacity_kwh) * 100.0

        current_V_pack = final_voltage_pack

        # --- Append Results ---
        sim_pv_to_load_ac.append(round(float(cur_pv_to_load), 3))
        # ***MODIFIED: Use the calculated DC power***
        sim_pv_to_battery_dc.append(round(float(dc_charge_from_pv_kw), 3))
        sim_pv_to_grid_ac.append(round(float(cur_pv_to_grid), 3))
        sim_battery_to_load_ac.append(round(float(cur_battery_to_load_ac), 3))
        sim_battery_to_grid_ac.append(round(float(cur_battery_to_grid_ac), 3))
        sim_grid_to_load_ac.append(round(float(cur_grid_to_load), 3))
        sim_grid_to_battery_ac.append(round(float(cur_grid_to_battery_ac), 3))
        sim_grid_import_ac.append(round(float(cur_grid_import_total), 3))
        sim_grid_export_ac.append(round(float(cur_grid_export_total), 3))

        sim_battery_soc_kwh_tracker.append(round(float(current_soc_kwh), 3))
        sim_battery_voltage_output.append(round(float(current_V_pack), 3))
        sim_battery_current_output.append(round(float(final_current_pack), 3))
        sim_neg_conc_tracker.append(final_neg_conc)
        sim_time_tracker.append(sim_time_tracker[-1] + time_points[-1] if time_points.size > 0 else sim_time_tracker[-1])

    # --- Final Metrics Calculation ---
    total_demand_kwh = sum(demand_profile_list) * interval_hours
    total_pv_gen_ac = sum([p * inverter_efficiency for p in pv_forecast_list]) * interval_hours

    energy_pv_to_load = sum(sim_pv_to_load_ac) * interval_hours
    energy_pv_to_battery_dc = sum(sim_pv_to_battery_dc) * interval_hours
    energy_pv_to_battery_ac_equiv = energy_pv_to_battery_dc * inverter_efficiency

    self_consumption = ((energy_pv_to_load + energy_pv_to_battery_ac_equiv) / total_pv_gen_ac) * 100 if total_pv_gen_ac > 0 else 0
    self_consumption = round(max(0, min(100, self_consumption)), 2)

    energy_from_pv_to_load_ac = sum(sim_pv_to_load_ac) * interval_hours
    energy_from_battery_to_load_ac = sum(sim_battery_to_load_ac) * interval_hours

    self_sufficiency = ((energy_from_pv_to_load_ac + energy_from_battery_to_load_ac) / total_demand_kwh) * 100 if total_demand_kwh > 0 else 0
    self_sufficiency = round(max(0, min(100, self_sufficiency)), 2)

    return {
        "pv_to_load_ac": sim_pv_to_load_ac,
        "pv_to_battery_dc": sim_pv_to_battery_dc,
        "pv_to_grid_ac": sim_pv_to_grid_ac,
        "battery_to_load_ac": sim_battery_to_load_ac,
        "battery_to_grid_ac": sim_battery_to_grid_ac,
        "grid_to_load_ac": sim_grid_to_load_ac,
        "grid_to_battery_ac": sim_grid_to_battery_ac,
        "grid_import_ac": sim_grid_import_ac,
        "grid_export_ac": sim_grid_export_ac,
        "battery_soc_kwh": sim_battery_soc_kwh_tracker,
        "battery_voltage": sim_battery_voltage_output,
        "battery_current": sim_battery_current_output,
        "interval_hours": interval_hours,
        "inverter_efficiency": inverter_efficiency,
        "battery_capacity_kwh": battery_capacity_kwh,
        "battery_max_charge_rate_kw": battery_max_charge_rate_kw,
        "battery_max_discharge_rate_kw": battery_max_discharge_rate_kw,
        "battery_efficiency": battery_efficiency,
        "initial_soc": initial_soc,
        "demand_profile_list": demand_profile_list,
        "original_demand_profile_list": demand_profile_list,
        "original_pv_forecast_dc": pv_forecast_list,
        "battery_min_soc_kwh": battery_min_soc_kwh,
        "battery_max_soc_kwh": battery_max_soc_kwh,
        "neg_particle_concentration_mol_m3": sim_neg_conc_tracker,
        "time_points_sec": sim_time_tracker,
        "battery_soc_kwh_pybamm": sim_battery_soc_kwh_tracker,
        "battery_voltage_pybamm": sim_battery_voltage_output,
        "battery_current_pybamm": sim_battery_current_output,
        "self_consumption (%)": self_consumption,
        "self_sufficiency (%)": self_sufficiency
    }


def mpc_controller(
    current_soc_kwh: float,
    pv_forecast: np.ndarray,
    demand_forecast: np.ndarray,
    price_forecast: np.ndarray,
    params: Dict,
    forecast_horizon_intervals: int,
    current_global_interval: int
) -> float:
    """
    Model Predictive Control function to determine the optimal battery action for the NEXT interval.
    Prioritizes minimizing grid import cost and maximizing self-consumption,
    with an added objective for time-specific SOC targets.
    """
    horizon_len = min(forecast_horizon_intervals, len(pv_forecast))
    pv_forecast_horizon = pv_forecast[:horizon_len]
    demand_forecast_horizon = demand_forecast[:horizon_len]
    price_forecast_horizon = price_forecast[:horizon_len]

    inverter_efficiency = params["inverter_efficiency"]
    interval_hours = params["interval_hours"]
    grid_export_price_factor = params.get("grid_export_price_factor", 1.0)

    soc_target_kwh_time = params.get("soc_target_kwh_time", {})
    soc_target_weight = params.get("soc_target_weight", 0.0)


    def objective(battery_power_kw_horizon):
        charge_power_ac = np.maximum(0, -battery_power_kw_horizon)
        discharge_power_ac = np.maximum(0, battery_power_kw_horizon)

        net_power_balance_ac = (pv_forecast_horizon * inverter_efficiency) - demand_forecast_horizon - charge_power_ac + discharge_power_ac

        grid_import_power = np.maximum(0, -net_power_balance_ac)
        grid_export_power = np.maximum(0, net_power_balance_ac)

        import_cost = np.sum(grid_import_power * price_forecast_horizon)
        export_revenue = np.sum(grid_export_power * price_forecast_horizon * grid_export_price_factor)

        soc_target_penalty = 0.0
        soc_path = np.zeros(horizon_len + 1)
        soc_path[0] = current_soc_kwh
        for t_horizon in range(horizon_len):
            soc_change_kwh = 0
            if battery_power_kw_horizon[t_horizon] > 0:
                soc_change_kwh = - (battery_power_kw_horizon[t_horizon] * interval_hours / params['battery_efficiency'])
            elif battery_power_kw_horizon[t_horizon] < 0:
                soc_change_kwh = abs(battery_power_kw_horizon[t_horizon]) * interval_hours * params['battery_efficiency']
            soc_path[t_horizon+1] = soc_path[t_horizon] + soc_change_kwh


        for target_hour_str, target_kwh in soc_target_kwh_time.items():
            try:
                target_hour = int(target_hour_str)
                target_global_interval = target_hour * 4

                if current_global_interval <= target_global_interval < (current_global_interval + horizon_len + 1):
                    index_in_soc_path = target_global_interval - current_global_interval

                    if 0 <= index_in_soc_path <= horizon_len:
                        predicted_soc_at_target_time = soc_path[index_in_soc_path]
                        deviation = predicted_soc_at_target_time - target_kwh
                        soc_target_penalty += soc_target_weight * (deviation ** 2)
            except ValueError:
                print(f"Warning: Invalid hour format in soc_target_kwh_time: {target_hour_str}")
                continue


        total_cost = (import_cost - export_revenue) * interval_hours + soc_target_penalty
        return total_cost

    def soc_lower_bound_constraint(battery_power_kw_horizon):
        soc_path_in_horizon = np.zeros(horizon_len + 1)
        soc_path_in_horizon[0] = current_soc_kwh
        for t_horizon in range(horizon_len):
            soc_change_kwh = 0
            if battery_power_kw_horizon[t_horizon] > 0:
                soc_change_kwh = - (battery_power_kw_horizon[t_horizon] * interval_hours / params['battery_efficiency'])
            elif battery_power_kw_horizon[t_horizon] < 0:
                soc_change_kwh = abs(battery_power_kw_horizon[t_horizon]) * interval_hours * params['battery_efficiency']
            soc_path_in_horizon[t_horizon+1] = soc_path_in_horizon[t_horizon] + soc_change_kwh
        return soc_path_in_horizon - params['battery_min_soc_kwh']

    def soc_upper_bound_constraint(battery_power_kw_horizon):
        soc_path_in_horizon = np.zeros(horizon_len + 1)
        soc_path_in_horizon[0] = current_soc_kwh
        for t_horizon in range(horizon_len):
            soc_change_kwh = 0
            if battery_power_kw_horizon[t_horizon] > 0:
                soc_change_kwh = - (battery_power_kw_horizon[t_horizon] * interval_hours / params['battery_efficiency'])
            elif battery_power_kw_horizon[t_horizon] < 0:
                soc_change_kwh = abs(battery_power_kw_horizon[t_horizon]) * interval_hours * params['battery_efficiency']
            soc_path_in_horizon[t_horizon+1] = soc_path_in_horizon[t_horizon] + soc_change_kwh
        return params['battery_max_soc_kwh'] - soc_path_in_horizon

    bounds = [(-params['battery_max_charge_rate_kw'], params['battery_max_discharge_rate_kw'])] * horizon_len

    initial_guess = np.zeros(horizon_len)

    constraints = [
        {'type': 'ineq', 'fun': soc_lower_bound_constraint},
        {'type': 'ineq', 'fun': soc_upper_bound_constraint}
    ]

    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    optimal_action = result.x[0]

    return float(optimal_action)


def mpc_simulation(
    pv_forecast_list: List[float],
    demand_profile_list: List[float],
    price_forecast_list: List[float],
    params: Dict
) -> Dict:
    """
    Runs a full-day simulation using the Model Predictive Control strategy.
    """
    n_slots = len(pv_forecast_list)
    interval_hours = params["interval_hours"]
    forecast_horizon_intervals = params["forecast_horizon_intervals"]

    local_battery_sim = BatterySimulator(
        Ns=params['Ns'],
        Np=params['Np'],
        V_cell_nominal=params['V_cell_nominal']
    )
    initial_x_neg = local_battery_sim.x_0 + (local_battery_sim.x_100 - local_battery_sim.x_0) * (params["initial_soc"] / params["battery_capacity_kwh"])

    current_soc_kwh = params["initial_soc"]
    current_soc_perc = (params["initial_soc"] / params["battery_capacity_kwh"]) * 100
    current_V_pack = params['V_cell_nominal'] * params['Ns']

    results = {
        "pv_to_load_ac": [], "grid_import_ac": [], "grid_export_ac": [],
        "battery_to_load_ac": [], "pv_to_battery_dc": [], "grid_to_battery_ac":[],
        "pv_to_grid_ac": [],
        "battery_to_grid_ac": [],
        "grid_to_load_ac": [],
        "battery_soc_kwh_pybamm": [current_soc_kwh], "hems_batt_charge_dc_kw": [],
        "hems_batt_discharge_dc_kw": [], "price_decisions": [],
        "neg_particle_concentration_mol_m3": [initial_x_neg],
        "time_points_sec": [0]
    }

    for i in range(n_slots):
        current_time_str = f"{int(i * interval_hours):02d}:{(int((i * interval_hours - int(i * interval_hours)) * 60)):02d}"

        remaining_pv = np.array(pv_forecast_list[i:])
        remaining_demand = np.array(demand_profile_list[i:])
        remaining_price = np.array(price_forecast_list[i:])

        optimal_battery_power_ac = mpc_controller(
            current_soc_kwh,
            remaining_pv,
            remaining_demand,
            remaining_price,
            params,
            forecast_horizon_intervals,
            current_global_interval=i
        )

        power_for_pybamm_kw = optimal_battery_power_ac
        if power_for_pybamm_kw > 0:
            power_for_pybamm_kw_adjusted = power_for_pybamm_kw / params['battery_efficiency']
        elif power_for_pybamm_kw < 0:
            power_for_pybamm_kw_adjusted = power_for_pybamm_kw * params['battery_efficiency']
        else:
            power_for_pybamm_kw_adjusted = 0.0

        final_soc_perc, final_V_pack, _, _, time_points, final_neg_conc = local_battery_sim.run_single_interval_simulation(
            power_for_pybamm_kw_adjusted,
            current_soc_perc / 100.0,
            current_V_pack
        )

        current_soc_perc = final_soc_perc
        current_soc_kwh = (current_soc_perc / 100.0) * params["battery_capacity_kwh"]
        current_V_pack = final_V_pack

        pv_ac_now = pv_forecast_list[i] * params['inverter_efficiency']
        demand_now = demand_profile_list[i]

        cur_pv_to_load = 0.0
        cur_battery_to_load = 0.0
        cur_grid_to_load = 0.0
        cur_pv_to_battery_dc = 0.0
        cur_grid_to_battery_ac = 0.0
        cur_pv_to_grid = 0.0
        cur_battery_to_grid = 0.0
        cur_grid_import_total = 0.0
        cur_grid_export_total = 0.0

        actual_pv_to_load = min(pv_ac_now, demand_now)
        cur_pv_to_load = actual_pv_to_load
        net_demand = demand_now - actual_pv_to_load
        pv_remaining = pv_ac_now - actual_pv_to_load

        if optimal_battery_power_ac < -0.01:
            charge_needed_ac = abs(optimal_battery_power_ac)

            charge_from_pv_ac = min(pv_remaining, charge_needed_ac)
            cur_pv_to_battery_dc = charge_from_pv_ac / params['inverter_efficiency']
            pv_remaining -= charge_from_pv_ac
            remaining_charge_needed_ac = charge_needed_ac - charge_from_pv_ac

            if remaining_charge_needed_ac > 0.01:
                cur_grid_to_battery_ac = remaining_charge_needed_ac
                cur_grid_import_total += cur_grid_to_battery_ac

        elif optimal_battery_power_ac > 0.01:
            discharge_available_ac = optimal_battery_power_ac

            discharge_to_load_from_battery = min(net_demand, discharge_available_ac)
            cur_battery_to_load = discharge_to_load_from_battery
            net_demand -= discharge_to_load_from_battery
            remaining_discharge_available_ac = discharge_available_ac - discharge_to_load_from_battery

            if remaining_discharge_available_ac > 0.01:
                if pv_ac_now < 0.01:
                    cur_battery_to_grid = 0.0
                else:
                    cur_battery_to_grid = remaining_discharge_available_ac

        if net_demand > 0.01:
            cur_grid_to_load = net_demand
            cur_grid_import_total += cur_grid_to_load

        if pv_remaining > 0.01:
            cur_pv_to_grid = pv_remaining

        cur_grid_export_total = cur_pv_to_grid + cur_battery_to_grid

        if cur_grid_import_total > 0.01 and cur_grid_export_total > 0.01:
            print(f"Warning: Simultaneous grid import ({cur_grid_import_total:.2f}) and export ({cur_grid_export_total:.2f}) at time {current_time_str}. Netting out.")
            net_flow = cur_grid_export_total - cur_grid_import_total
            if net_flow > 0:
                cur_grid_export_total = net_flow
                cur_grid_import_total = 0.0
            else:
                cur_grid_import_total = abs(net_flow)
                cur_grid_export_total = 0.0

        results["pv_to_load_ac"].append(round(float(cur_pv_to_load), 3))
        results["battery_to_load_ac"].append(round(float(cur_battery_to_load), 3))
        results["grid_to_load_ac"].append(round(float(cur_grid_to_load), 3))
        results["pv_to_battery_dc"].append(round(float(cur_pv_to_battery_dc), 3))
        results["grid_to_battery_ac"].append(round(float(cur_grid_to_battery_ac), 3))
        results["pv_to_grid_ac"].append(round(float(cur_pv_to_grid), 3))
        results["battery_to_grid_ac"].append(round(float(cur_battery_to_grid), 3))
        results["grid_import_ac"].append(round(float(cur_grid_import_total), 3))
        results["grid_export_ac"].append(round(float(cur_grid_export_total), 3))

        results["battery_soc_kwh_pybamm"].append(round(float(current_soc_kwh), 3))
        results["hems_batt_charge_dc_kw"].append(round(float(max(0, -optimal_battery_power_ac)), 3))
        results["hems_batt_discharge_dc_kw"].append(round(float(max(0, optimal_battery_power_ac)), 3))
        results["price_decisions"].append(round(float(optimal_battery_power_ac), 3))
        results["neg_particle_concentration_mol_m3"].append(final_neg_conc)
        results["time_points_sec"].append(results["time_points_sec"][-1] + (time_points[-1] if time_points.size > 0 else 900))

    total_demand_kwh = sum(demand_profile_list) * interval_hours
    total_pv_gen_ac = sum([p * params["inverter_efficiency"] for p in pv_forecast_list]) * interval_hours

    energy_pv_to_load = sum(results["pv_to_load_ac"]) * interval_hours
    energy_pv_to_battery_dc = sum(results["pv_to_battery_dc"]) * interval_hours
    energy_pv_to_battery_ac_equiv = energy_pv_to_battery_dc * params["inverter_efficiency"]

    self_consumption = ((energy_pv_to_load + energy_pv_to_battery_ac_equiv) / total_pv_gen_ac) * 100 if total_pv_gen_ac > 0 else 0
    self_consumption = round(max(0, min(100, self_consumption)), 2)

    energy_from_pv_to_load_ac = sum(results["pv_to_load_ac"]) * interval_hours
    energy_from_battery_to_load_ac = sum(results["battery_to_load_ac"]) * interval_hours

    self_sufficiency = ((energy_from_pv_to_load_ac + energy_from_battery_to_load_ac) / total_demand_kwh) * 100 if total_demand_kwh > 0 else 0
    self_sufficiency = round(max(0, min(100, self_sufficiency)), 2)
    results.update({
        "interval_hours": interval_hours, "inverter_efficiency": params['inverter_efficiency'],
        "battery_capacity_kwh": params['battery_capacity_kwh'], "battery_max_charge_rate_kw": params['battery_max_charge_rate_kw'],
        "battery_max_discharge_rate_kw": params['battery_max_discharge_rate_kw'], "battery_efficiency": params['battery_efficiency'],
        "initial_soc": params['initial_soc'], "original_demand_profile_list": demand_profile_list,
        "original_pv_forecast_dc": pv_forecast_list, "battery_min_soc_kwh": params['battery_min_soc_kwh'],
        "battery_max_soc_kwh": params['battery_max_soc_kwh'],
        "self_consumption (%)": self_consumption,
        "self_sufficiency (%)": self_sufficiency
    })

    return results


def calculate_daily_bill(
    grid_import_ac: List[float],
    grid_export_ac: List[float],
    price_profile: List[float],
    interval_hours: float = 0.25
) -> Dict[str, float]:
    """
    Calculates the total daily electricity bill based on grid import, export, and prices.
    """
    total_import_cost = 0.0
    total_export_revenue = 0.0
    n_intervals = len(grid_import_ac)

    min_len = min(n_intervals, len(grid_export_ac), len(price_profile))

    grid_import_ac = grid_import_ac[:min_len]
    grid_export_ac = grid_export_ac[:min_len]
    price_profile = price_profile[:min_len]


    for i in range(min_len):
        import_kwh_this_interval = grid_import_ac[i] * interval_hours
        export_kwh_this_interval = grid_export_ac[i] * interval_hours
        price_this_interval = price_profile[i]

        total_import_cost += import_kwh_this_interval * price_this_interval
        total_export_revenue += export_kwh_this_interval * price_this_interval

    net_bill = total_import_cost - total_export_revenue

    return {
        "total_import_cost_rs": round(total_import_cost, 3),
        "net_bill_rs": round(net_bill, 3)
    }

# New Function: Save results to Excel ---
def save_simulation_results_to_excel(
    results: Dict,
    pv_data: List[float],
    demand_data: List[float],
    price_data: List[float],
    file_path: str
):
    """
    Saves comprehensive simulation results to a single Excel file.
    """
    n_slots = len(pv_data)
    interval_hours = results.get("interval_hours", 0.25)

    time_index = pd.to_datetime([f"{int(h):02d}:{(int((h - int(h)) * 60)):02d}" for h in np.arange(0, n_slots * interval_hours, interval_hours)]).time

    data = {
        'Time': time_index,
        'PV Forecast (kW_dc)': pv_data,
        'Appliance Demand (kW_ac)': demand_data,
        'Electricity Price (Rs/kWh)': price_data,
        'PV to Load (kW_ac)': results.get("pv_to_load_ac", []),
        'PV to Battery (kW_dc)': results.get("pv_to_battery_dc", []),
        'PV to Grid (kW_ac)': results.get("pv_to_grid_ac", []),
        'Grid to Battery (kW_ac)': results.get("grid_to_battery_ac", []),
        'Battery to Load (kW_ac)': results.get("battery_to_load_ac", []),
        'Battery to Grid (kW_ac)': results.get("battery_to_grid_ac", []),
        'Grid to Load (kW_ac)': results.get("grid_to_load_ac", []),
        'Total Grid Import (kW_ac)': results.get("grid_import_ac", []),
        'Total Grid Export (kW_ac)': results.get("grid_export_ac", [])
    }

    if "price_decisions" in results:
        data['MPC Battery Power (kW)'] = results["price_decisions"]


    soc_data_kwh = results.get("battery_soc_kwh_pybamm", results.get("battery_soc_kwh", []))

    df = pd.DataFrame(data)

    if len(soc_data_kwh) > n_slots:
        df['Battery SOC (kWh)'] = soc_data_kwh[1:]
    else:
        df['Battery SOC (kWh)'] = soc_data_kwh


    try:
        output_dir = os.path.dirname(file_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        df.to_excel(file_path, index=False)
        print(f"Simulation results successfully saved to {file_path}")
        return {"status": "success", "message": f"Simulation results saved to {file_path}"}
    except Exception as e:
        print(f"Error saving results to Excel: {e}")
        return {"status": "error", "message": f"Error saving results to Excel: {e}"}


# --- NEW: Appliance Control and Suggestion Functions ---
def control_appliance(appliance_name: str, action: str, preferred_time: float = None) -> dict:
    """
    Simulates controlling an appliance. In a real system, this would interact with smart plugs/devices.
    For now, it's a placeholder to demonstrate LLM interaction.
    Args:
        appliance_name (str): The name of the appliance (e.g., "Washing Machine").
        action (str): The desired action ("turn on", "turn off", "schedule").
        preferred_time (float, optional): The preferred hour (e.g., 8.5 for 8:30 AM) for scheduling.
    Returns:
        dict: Status message.
    """
    status_message = f"Attempted to {action} {appliance_name}"
    if preferred_time is not None:
        status_message += f" at {preferred_time:.2f} hours."
    else:
        status_message += "."
    print(f"Appliance Control: {status_message}")
    return {"status": "success", "message": status_message}

def suggest_optimal_appliance_time(appliance_name: str, date: str = None,
                                   pv_forecast_list: List[float] = None,
                                   price_profile_list: List[float] = None,
                                   individual_appliance_power_per_slot: Dict[str, List[float]] = None,
                                   appliances_data: List[Dict] = None) -> dict:
    """
    Suggests an optimal time for an appliance based on current forecasts (PV, demand, price)
    to minimize total daily cost.
    Args:
        appliance_name (str): The name of the appliance.
        date (str, optional): The date for which to suggest the time. Defaults to today.
        pv_forecast_list (List[float], optional): List of PV generation forecasts (kW_dc) for the day.
        price_profile_list (List[float], optional): List of electricity prices (Rs/kWh) for the day.
        individual_appliance_power_per_slot (Dict[str, List[float]], optional): Dictionary of
            individual appliance power profiles for the day, *including the target appliance's default schedule*.
        appliances_data (List[Dict], optional): List of all appliance data.
    Returns:
        dict: A suggested time or error message.
    """
    print(f"DEBUG (suggest_optimal_appliance_time): Received appliance_name: '{appliance_name}'")

    if date is None:
        date = datetime.date.today().strftime("%Y-%m-%d")

    if appliances_data is None:
        appliances_data = DEFAULT_APPLIANCES_DATA

    appliance_info = next((app for app in appliances_data if app["name"] == appliance_name), None)
    print(f"DEBUG (suggest_optimal_appliance_time): appliance_info after lookup: {appliance_info}")

    if not appliance_info:
        print(f"DEBUG (suggest_optimal_appliance_time): Appliance info not found for '{appliance_name}'. Returning error.")
        return {"status": "error", "message": f"Appliance '{appliance_name}' not found in the defined list."}

    if "power" not in appliance_info or "duration" not in appliance_info:
        print(f"DEBUG (suggest_optimal_appliance_time): Appliance info missing 'power' or 'duration' for '{appliance_name}'. Appliance info: {appliance_info}. Returning error.")
        return {"status": "error", "message": f"Appliance '{appliance_name}' data is incomplete (missing 'power' or 'duration')."}

    n_slots = 96
    interval_hours = 0.25
    inverter_efficiency = 0.95

    if len(pv_forecast_list) != n_slots or len(price_profile_list) != n_slots:
        return {"status": "error", "message": "Inconsistent forecast data length. Expected 96 intervals for PV and price."}

    for app_name, profile in individual_appliance_power_per_slot.items():
        if len(profile) != n_slots:
            return {"status": "error", "message": f"Individual appliance profile for {app_name} has inconsistent length ({len(profile)}). Expected {n_slots} intervals."}


    appliance_power_kw = appliance_info["power"] * 1000.0
    appliance_duration_slots = int(appliance_info["duration"] * 4)
    print(f"DEBUG (suggest_optimal_appliance_time): Appliance '{appliance_name}' power={appliance_power_kw}, duration_slots={appliance_duration_slots}")


    try:
        current_date_obj = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        return {"status": "error", "message": f"Invalid date format: {date}. Please useYYYY-MM-DD."}

    is_weekend = current_date_obj.weekday() >= 5
    schedules = appliance_info.get("schedule_weekend" if is_weekend else "schedule_weekday", [])

    if not schedules:
        return {"status": "info", "message": f"No specific schedule found for {appliance_name} on {date}. Cannot provide an optimized time based on schedules."}

    best_total_daily_cost = float('inf')
    optimal_start_hour = None

    base_demand_without_target_appliance = np.zeros(n_slots)
    for other_appliance_name, other_app_profile in individual_appliance_power_per_slot.items():
        if other_appliance_name != appliance_name:
            base_demand_without_target_appliance += np.array(other_app_profile)

    for schedule in schedules:
        start_slot_window = int(schedule["start"] * 4)
        end_slot_window = int(schedule["end"] * 4)
        print(f"DEBUG (suggest_optimal_appliance_time): Schedule window for '{appliance_name}': start_slot={start_slot_window}, end_slot={end_slot_window}")
        print(f"DEBUG (suggest_optimal_appliance_time): Appliance duration slots: {appliance_duration_slots}")

        max_candidate_start_slot = end_slot_window - appliance_duration_slots
        print(f"DEBUG (suggest_optimal_appliance_time): Max candidate start slot: {max_candidate_start_slot}")

        if max_candidate_start_slot < start_slot_window:
            print(f"DEBUG (suggest_optimal_appliance_time): Warning: Schedule window too small for appliance duration. Skipping this schedule entry. Start: {start_slot_window}, End: {end_slot_window}, Duration: {appliance_duration_slots}")
            continue

        for current_slot in range(start_slot_window, max_candidate_start_slot + 1):
            if current_slot < 0 or current_slot + appliance_duration_slots > n_slots:
                print(f"DEBUG (suggest_optimal_appliance_time): Skipping candidate_start_slot {current_slot} due to boundary conditions.")
                continue

            hypothetical_total_demand = np.array(base_demand_without_target_appliance).copy()

            for i in range(appliance_duration_slots):
                slot_to_add = current_slot + i
                if slot_to_add < n_slots:
                    hypothetical_total_demand[slot_to_add] += appliance_power_kw

            sim_grid_import_ac = []
            sim_grid_export_ac = []

            for s in range(n_slots):
                pv_ac_at_slot = pv_forecast_list[s] * inverter_efficiency
                total_demand_at_slot = hypothetical_total_demand[s]
                price_at_slot = price_profile_list[s]

                net_load = total_demand_at_slot - pv_ac_at_slot

                if net_load > 0:
                    sim_grid_import_ac.append(net_load)
                    sim_grid_export_ac.append(0.0)
                else:
                    sim_grid_import_ac.append(0.0)
                    sim_grid_export_ac.append(abs(net_load))

            current_day_bill_info = calculate_daily_bill(
                grid_import_ac=sim_grid_import_ac,
                grid_export_ac=sim_grid_export_ac,
                price_profile=price_profile_list,
                interval_hours=interval_hours
            )

            current_total_daily_cost = current_day_bill_info["net_bill_rs"]

            if current_total_daily_cost < best_total_daily_cost:
                best_total_daily_cost = current_total_daily_cost
                optimal_start_hour = current_slot * interval_hours

    if optimal_start_hour is not None:
        suggested_time_str = f"{int(optimal_start_hour):02d}:{(int((optimal_start_hour - int(optimal_start_hour)) * 60)):02d}"
        message = (f"For {appliance_name} on {date}, the optimal time to run it is around {suggested_time_str} "
                   f"to achieve an estimated minimal daily cost of Rs. {best_total_daily_cost:.2f}. "
                   "This accounts for maximizing PV self-consumption and minimizing grid import costs across the entire day.")
        print(f"Appliance Suggestion: {message}")
        return {"status": "success", "message": message, "suggested_time_rationale": "Optimized based on PV generation and electricity prices for minimal total daily cost."}
    else:
        message = f"Could not determine an optimal time for {appliance_name} on {date} within its schedules. Please check appliance data, forecast data, or schedules."
        print(f"Appliance Suggestion: {message}")
        return {"status": "info", "message": message, "suggested_time_rationale": "No suitable time found within constraints."}


# NEW FUNCTION: suggest_optimal_appliance_time_with_mpc (with result saving)
def suggest_optimal_appliance_time_with_mpc(appliance_name: str, date: str = None,
                                             system_params: Dict = None,
                                             appliances_data: List[Dict] = None,
                                             solar_excel_file_path: str = None,
                                             demand_data_file_path: str = None) -> dict:
    """
    Suggests an optimal time for a controllable appliance by integrating its schedule into the overall MPC-managed energy system to minimize the total daily cost.
    This function will run MPC simulations for each candidate time slot and save the best result to an Excel file.
    """
    print(f"DEBUG (suggest_optimal_appliance_time_with_mpc): Received appliance_name: '{appliance_name}', date: '{date}', solar_excel_file_path: '{solar_excel_file_path}', demand_data_file_path: '{demand_data_file_path}'")

    if date is None:
        date = datetime.date.today().strftime("%Y-%m-%d")

    if system_params is None:
        system_params = {
            "battery_capacity_kwh": 10.24, "initial_soc": 5.12,
            "battery_min_soc_kwh": 0.2 * 10.24, "battery_max_soc_kwh": 0.8 * 10.24,
            "battery_max_charge_rate_kw": 3.0, "battery_max_discharge_rate_kw": 3.0,
            "battery_efficiency": 0.95, "inverter_efficiency": 0.95,
            "interval_hours": 0.25, 'Ns': 60, 'Np': 10, 'V_cell_nominal': 3.65,
            "forecast_horizon_intervals": 32, "grid_export_price_factor": 0.01,
            "soc_target_kwh_time": {"18": 7.2}, "soc_target_weight": 50.0,
            "forecast_origin_intervals": 10
        }

    if appliances_data is None:
        appliances_data = DEFAULT_APPLIANCES_DATA

    appliance_info = next((app for app in appliances_data if app["name"] == appliance_name), None)
    if not appliance_info:
        return {"status": "error", "message": f"Appliance '{appliance_name}' not found in the defined list."}
    if "power" not in appliance_info or "duration" not in appliance_info:
        return {"status": "error", "message": f"Appliance '{appliance_name}' data is incomplete (missing 'power' or 'duration')."}

    n_slots = 96
    interval_hours = system_params["interval_hours"]
    appliance_power_kw = appliance_info["power"] / 1000.0
    appliance_duration_slots = int(appliance_info["duration"] * 4)

    try:
        current_date_obj = datetime.datetime.strptime(date, "%Y-%m-%d").date()
        pv_forecast_output = generate_pv_forecast(solar_excel=solar_excel_file_path, selected_day=date)
        if "error" in pv_forecast_output:
            return {"status": "error", "message": f"Failed to get PV forecast: {pv_forecast_output['error']}"}
        pv_forecast_list = pv_forecast_output["dc_power_forecast"]

        price_forecast_output = get_iex_mcp(date_str=current_date_obj.strftime("%d-%m-%Y"))
        if "error" in price_forecast_output:
            return {"status": "error", "message": f"Failed to fetch prices: {price_forecast_output['error']}"}
        price_profile_list = price_forecast_output["price_profile"]

        if len(pv_forecast_list) != n_slots or len(price_profile_list) != n_slots:
            return {"status": "error", "message": "Inconsistent forecast data length. Expected 96 intervals for PV and price."}
    except Exception as e:
        return {"status": "error", "message": f"Error fetching essential data for MPC scheduling: {e}"}

    is_weekend = current_date_obj.weekday() >= 5
    schedules = appliance_info.get("schedule_weekend" if is_weekend else "schedule_weekday", [])
    if not schedules:
        return {"status": "info", "message": f"No specific schedule found for {appliance_name} on {date}."}

    try:
        print(f"DEBUG: Attempting to load demand data from: {demand_data_file_path}")
        if not os.path.exists(demand_data_file_path):
            return {"status": "error", "message": f"Demand data file not found at: {demand_data_file_path}."}
        full_demand_df = pd.read_excel(demand_data_file_path)
        full_demand_df["DateTime"] = pd.to_datetime(full_demand_df["Date"].astype(str) + " " + full_demand_df["Time"].astype(str), errors="coerce")
        full_demand_df = full_demand_df.dropna(subset=["DateTime"]).set_index("DateTime")
        daily_demand_df_for_date = full_demand_df[(full_demand_df.index.date == current_date_obj)].sort_index()
        if daily_demand_df_for_date.empty:
            return {"status": "error", "message": f"No demand data found for {date} in {demand_data_file_path}."}

        all_appliance_columns_by_index = daily_demand_df_for_date.columns[2:18].tolist()
        columns_to_sum_for_base_demand = [col for col in all_appliance_columns_by_index if col != appliance_name]

        if not columns_to_sum_for_base_demand:
            base_demand_without_target_appliance = np.zeros(n_slots)
        else:
            base_demand_without_target_appliance = daily_demand_df_for_date[columns_to_sum_for_base_demand].sum(axis=1).head(n_slots).to_numpy() #* 1000.0
            base_demand_without_target_appliance = np.maximum(0, base_demand_without_target_appliance)
    except Exception as e:
        return {"status": "error", "message": f"Error loading or processing demand data from {demand_data_file_path}: {e}"}

    best_total_daily_cost_with_appliance = float('inf')
    optimal_start_hour = None
    best_mpc_results = None
    best_demand_profile = None

    for schedule_window in schedules:
        start_slot_window = int(schedule_window["start"] * 4)
        end_slot_window = int(schedule_window["end"] * 4)
        max_candidate_start_slot = end_slot_window - appliance_duration_slots
        if max_candidate_start_slot < start_slot_window:
            continue

        for current_slot in range(start_slot_window, max_candidate_start_slot + 1):
            if current_slot < 0 or current_slot + appliance_duration_slots > n_slots:
                continue

            hypothetical_total_demand = np.array(base_demand_without_target_appliance).copy()
            for i in range(appliance_duration_slots):
                slot_to_add = current_slot + i
                if slot_to_add < n_slots:
                    hypothetical_total_demand[slot_to_add] += appliance_power_kw

            current_mpc_results = mpc_simulation(
                pv_forecast_list=pv_forecast_list,
                demand_profile_list=hypothetical_total_demand.tolist(),
                price_forecast_list=price_profile_list,
                params=system_params
            )
            if "error" in current_mpc_results:
                print(f"Warning: MPC simulation failed for candidate slot {current_slot}: {current_mpc_results.get('message', 'Unknown error')}. Skipping.")
                continue

            current_day_bill_info = calculate_daily_bill(
                grid_import_ac=current_mpc_results["grid_import_ac"],
                grid_export_ac=current_mpc_results["grid_export_ac"],
                price_profile=price_profile_list,
                interval_hours=interval_hours
            )
            current_total_daily_cost = current_day_bill_info["net_bill_rs"]

            if current_total_daily_cost < best_total_daily_cost_with_appliance:
                best_total_daily_cost_with_appliance = current_total_daily_cost
                optimal_start_hour = current_slot * interval_hours
                best_mpc_results = current_mpc_results
                best_demand_profile = hypothetical_total_demand.tolist()

        # Check if any optimal hour was found in this schedule window
        if optimal_start_hour is None:
            print(f"DEBUG: No optimal start hour found within schedule {schedule_window}. This may be an error if the loop should have run.")
            continue # Go to the next schedule window

        # Ensure best_demand_profile is not None before proceeding
        if best_demand_profile is None:
            # This case might happen if all MPC sims failed, but an optimal hour was somehow set.
            # As a fallback, use the last hypothetical demand.
            print("Warning: best_demand_profile is None after schedule loop. Using last hypothetical demand.")
            best_demand_profile = hypothetical_total_demand.tolist() if 'hypothetical_total_demand' in locals() else [0.0] * n_slots


        total_demand_kwh = sum(best_demand_profile) * interval_hours
        total_pv_gen_ac = sum([p * system_params["inverter_efficiency"] for p in pv_forecast_list]) * interval_hours

        energy_pv_to_load = sum(best_mpc_results["pv_to_load_ac"]) * interval_hours
        energy_pv_to_battery_dc = sum(best_mpc_results["pv_to_battery_dc"]) * interval_hours
        energy_pv_to_battery_ac_equiv = energy_pv_to_battery_dc * system_params["inverter_efficiency"]

        self_consumption = ((energy_pv_to_load + energy_pv_to_battery_ac_equiv) / total_pv_gen_ac) * 100 if total_pv_gen_ac > 0 else 0
        self_consumption = round(max(0, min(100, self_consumption)), 2)

        energy_from_pv_to_load_ac = sum(best_mpc_results["pv_to_load_ac"]) * interval_hours
        energy_from_battery_to_load_ac = sum(best_mpc_results["battery_to_load_ac"]) * interval_hours

        self_sufficiency = ((energy_from_pv_to_load_ac + energy_from_battery_to_load_ac) / total_demand_kwh) * 100 if total_demand_kwh > 0 else 0
        self_sufficiency = round(max(0, min(100, self_sufficiency)), 2)

        # Update the best_mpc_results with the calculated metrics
        best_mpc_results["self_consumption (%)"] = self_consumption
        best_mpc_results["self_sufficiency (%)"] = self_sufficiency


    if optimal_start_hour is not None and best_mpc_results is not None:
        suggested_time_str = f"{int(optimal_start_hour):02d}:{(int((optimal_start_hour - int(optimal_start_hour)) * 60)):02d}"

        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)

        file_path = os.path.join(output_dir, f"optimal_appliance_schedule_{appliance_name.replace(' ', '_')}_{date}.xlsx")

        # Ensure best_demand_profile is valid before saving
        if best_demand_profile is None:
             best_demand_profile = [0.0] * n_slots # Fallback
             print("ERROR: best_demand_profile was None before saving. Using empty list.")


        save_simulation_results_to_excel(
            results=best_mpc_results,
            pv_data=pv_forecast_list,
            demand_data=best_demand_profile,
            price_data=price_profile_list,
            file_path=file_path
        )

        self_consumption = best_mpc_results.get('self_consumption (%)', 0)
        self_sufficiency = best_mpc_results.get('self_sufficiency (%)', 0)

        message = (f"For {appliance_name} on {date}, considering the overall MPC-managed system, "
                   f"the optimal time to run it is around {suggested_time_str} "
                   f"to achieve an estimated minimal daily cost of Rs. {best_total_daily_cost_with_appliance:.2f}. "
                   f"With this schedule, the estimated Self-Consumption is {self_consumption:.2f}% and Self-Sufficiency is {self_sufficiency:.2f}%. "
                   f"A detailed breakdown of this optimal schedule has been saved to '{file_path}'.")

        print(f"Appliance Suggestion (MPC-aware): {message}")
        return {"status": "success", "message": message, "saved_results_path": file_path}
    else:
        message = f"Could not determine an optimal time for {appliance_name} on {date} within its schedules, even with MPC consideration."
        print(f"Appliance Suggestion (MPC-aware): {message}")
        return {"status": "info", "message": message}

def run_conversation_with_direct_tool_call(
    user_query: str, solar_data_file_path: str, demand_data_file_path: str,
    system_params: Dict = None
):
    """
    Runs the conversation with the API-based tool caller, now handling all tools.
    """
    if system_params is None:
        system_params = {
            "battery_capacity_kwh": 10.24, "initial_soc": 5.12, "battery_min_soc_kwh": 2.048,
            "battery_max_soc_kwh": 8.192, "battery_max_charge_rate_kw": 3.0, "battery_max_discharge_rate_kw": 3.0,
            "battery_efficiency": 0.95, "inverter_efficiency": 0.95, "interval_hours": 0.25,
            'Ns': 60, 'Np': 10, 'V_cell_nominal': 3.65, "forecast_horizon_intervals": 32,
            "grid_export_price_factor": 0.01
        }

    response_data = {"query": user_query, "tool_used": "N/A", "final_ai_response": "Could not process request."}

    parsed_llm_output = call_gemini_for_tool(user_query)
    tool_name = parsed_llm_output.get("tool_name", "N/A")
    tool_params = parsed_llm_output.get("parameters", {})
    response_data["tool_used"] = tool_name

    # If the LLM fails to extract a date, try to find one with regex.
    if "date" not in tool_params:
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', user_query)
        if date_match:
            print(f"INFO: LLM failed to extract date. Using regex fallback: {date_match.group(1)}")
            tool_params["date"] = date_match.group(1)

    # --- Date Standardization ---
    today = datetime.date.today()
    if "date" in tool_params and isinstance(tool_params["date"], str):
        date_str = tool_params["date"].lower()
        if date_str == "today":
            tool_params["date"] = today.strftime("%Y-%m-%d")
        elif date_str == "tomorrow":
            tool_params["date"] = (today + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        elif date_str == "yesterday":
            tool_params["date"] = (today - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        elif date_str == "the day after tomorrow":
            tool_params["date"] = (today + datetime.timedelta(days=2)).strftime("%Y-%m-%d")

    tool_output = {"status": "error", "message": "Tool execution failed."}

    try:
        if tool_name in ["mpc_simulation", "baseline_reactive_price_blind", "heuristic_price_aware_baseline"]:
            sim_date = tool_params.get("date", today.strftime("%Y-%m-%d"))

            # Fetch PV data (needed by all)
            pv_forecast_result = generate_pv_forecast(solar_excel=solar_data_file_path, selected_day=sim_date)
            if "error" in pv_forecast_result: raise ValueError(f"PV forecast error: {pv_forecast_result['error']}")
            pv_data = pv_forecast_result["dc_power_forecast"]

            # Fetch Demand data (needed by all)
            demand_df = pd.read_excel(demand_data_file_path)
            demand_df.index = pd.to_datetime(demand_df['Date'].astype(str) + ' ' + demand_df['Time'].astype(str))
            demand_data = demand_df.loc[demand_df.index.date == pd.to_datetime(sim_date).date()]['Total Load'].tolist()
            if not demand_data: raise ValueError(f"No demand data found for {sim_date} in the provided file.")
            if len(demand_data) < 96:
                print(f"Warning: Incomplete demand data ({len(demand_data)}/96). Padding with last known value.")
                demand_data.extend([demand_data[-1]] * (96 - len(demand_data)))
            demand_data = demand_data[:96]


            # Fetch Price data (ONLY needed for mpc and heuristic)
            price_data = [0.0] * 96 # Default for price-blind
            if tool_name in ["mpc_simulation", "heuristic_price_aware_baseline"]:
                price_data_result = get_iex_mcp(date_str=pd.to_datetime(sim_date).strftime("%d-%m-%Y"))
                if "error" in price_data_result: raise ValueError(f"Price data error: {price_data_result['error']}")
                price_data = price_data_result["price_profile"]
                # Handle incomplete price data
                if len(price_data) < 96:
                    print(f"Warning: Incomplete price data ({len(price_data)}/96). Padding with last known price.")
                    price_data.extend([price_data[-1]] * (96 - len(price_data)))
                price_data = price_data[:96]


            # Run the correct simulation
            if tool_name == "mpc_simulation":
                tool_output = mpc_simulation(pv_data, demand_data, price_data, system_params)
            elif tool_name == "heuristic_price_aware_baseline":
                tool_output = heuristic_price_aware_baseline(pv_data, demand_data, price_data, system_params)
            else: # tool_name == "baseline_reactive_price_blind"
                tool_output = baseline_reactive_price_blind(pv_data, demand_data, system_params)

            bill_info = calculate_daily_bill(tool_output["grid_import_ac"], tool_output["grid_export_ac"], price_data)

            # --- NEW: Automatically save results to Excel ---
            output_dir = "results"
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, f"{tool_name}_results_{sim_date}.xlsx")

            save_status = save_simulation_results_to_excel(
                results=tool_output,
                pv_data=pv_data,
                demand_data=demand_data,
                price_data=price_data,
                file_path=file_path
            )

            save_message = f"Results saved to '{file_path}'." if save_status.get("status") == "success" else "Failed to save results to Excel."
            # --- End of new code ---

            response_data["final_ai_response"] = (
                f"Generated {tool_name} plan for {sim_date}. Net bill: Rs. {bill_info['net_bill_rs']:.2f}. "
                f"Self-Consumption: {tool_output.get('self_consumption (%)', 0):.2f}%, Self-Sufficiency: {tool_output.get('self_sufficiency (%)', 0):.2f}%. "
                f"{save_message}"
            )
            plot_simulation_results(tool_output, price_data, date_label=sim_date, title_prefix=tool_name.upper())

        elif tool_name == "suggest_optimal_appliance_time_with_mpc":
            if 'appliance' in tool_params and 'appliance_name' not in tool_params:
                tool_params['appliance_name'] = tool_params.pop('appliance')

            tool_output = suggest_optimal_appliance_time_with_mpc(
                **tool_params, system_params=system_params, solar_excel_file_path=solar_data_file_path,
                demand_data_file_path=demand_data_file_path)
            response_data["final_ai_response"] = tool_output.get("message", "Suggestion generated.")

        elif tool_name == "get_solar_forecast":
            date = tool_params.get("date", today.strftime("%Y-%m-%d"))
            tool_output = generate_pv_forecast(solar_excel=solar_data_file_path, selected_day=date)
            if "error" not in tool_output:
                total_kwh = sum(tool_output['dc_power_forecast']) * 0.25
                response_data['final_ai_response'] = f"Total forecasted PV generation for {date}: {total_kwh:.2f} kWh (DC)."
            else:
                response_data['final_ai_response'] = tool_output['error']

        elif tool_name == "plot_solar_forecast":
            date = tool_params.get("date", today.strftime("%Y-%m-%d"))
            tool_output = generate_pv_forecast(solar_excel=solar_data_file_path, selected_day=date)
            if "error" not in tool_output:
                pv_ac = [p * system_params['inverter_efficiency'] for p in tool_output['dc_power_forecast']]
                plot_solar_forecast(pv_ac, date_label=date)
                response_data['final_ai_response'] = f"Displaying the solar forecast plot for {date}."
            else:
                response_data['final_ai_response'] = tool_output['error']

        elif tool_name == "get_electricity_price":
            date = tool_params.get("date", today.strftime("%d-%m-%Y"))
            market = tool_params.get("market_type", "real-time")
            tool_output = get_iex_mcp(date_str=date, market_type=market)
            if "error" not in tool_output:
                avg_price = np.mean(tool_output['price_profile'])
                response_data['final_ai_response'] = f"The average {market} electricity price for {date} is Rs. {avg_price:.2f}/kWh."
            else:
                response_data['final_ai_response'] = tool_output['error']

        elif tool_name == "plot_electricity_price":
            date = tool_params.get("date", today.strftime("%d-%m-%Y"))
            market = tool_params.get("market_type", "real-time")
            tool_output = get_iex_mcp(date_str=date, market_type=market)
            if "error" not in tool_output:
                plot_single_price_profile(tool_output['price_profile'], date_label=date, market_type=market)
                response_data['final_ai_response'] = f"Displaying the {market} electricity price plot for {date}."
            else:
                response_data['final_ai_response'] = tool_output['error']

        elif tool_name == "control_appliance":
             tool_output = control_appliance(**tool_params)
             response_data['final_ai_response'] = tool_output.get("message", "Appliance control command issued.")

        elif tool_name == "N/A":
             response_data['final_ai_response'] = "I am an energy management assistant. Please ask me about solar forecasts, electricity prices, or energy planning."
             tool_output = {"status": "info", "message": "Out of scope query."}

        else:
            response_data['final_ai_response'] = f"The tool '{tool_name}' is recognized but not implemented in the conversation runner."
            tool_output = {"status": "error", "message": "Tool not implemented."}

    except Exception as e:
        response_data["final_ai_response"] = f"An error occurred while executing tool '{tool_name}': {e}"
        tool_output = {"status": "error", "message": str(e)}

    response_data["tool_output_summary"] = tool_output
    return response_data


if __name__ == '__main__':

    # --- 2. Configuration for Demonstrations ---
    solar_excel_file_path = '/content/drive/MyDrive/HEMS/pv_forecast_2024_4strings.xlsx'
    demand_data_file = '/content/drive/MyDrive/HEMS/appliances_data_daywise3.xlsx'
    solar_excel = '/content/drive/MyDrive/solar.xlsx'
    system_params = {
        "battery_capacity_kwh": 10.24, "initial_soc": 5.12, "battery_min_soc_kwh": 2.048,
        "battery_max_soc_kwh": 8.192, "battery_max_charge_rate_kw": 3.0, "battery_max_discharge_rate_kw": 3.0,
        "battery_efficiency": 0.95, "inverter_efficiency": 0.95, "interval_hours": 0.25,
        'Ns': 60, 'Np': 10, 'V_cell_nominal': 3.65, "forecast_horizon_intervals": 32,
        "grid_export_price_factor": 0.01
    }

    run_conversation_with_gemini = run_conversation_with_direct_tool_call
    start_time = time.time()
    print("\n" + "="*80 + "\n--- Running Demonstration: MPC Simulation ---")
    user_input_1 = "plan energy scheduling for 2024-05-06"
    response_1 = run_conversation_with_gemini(user_input_1, solar_excel, demand_data_file, system_params)
    print(json.dumps(response_1, indent=2))
    end_time= time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    print("\n" + "="*80 + "\n--- Running Demonstration: Baseline (Reactive, Price-Blind) Simulation ---")
    user_input_2 = "plan baseline energy schedule for 2024-05-06."
    response_2 = run_conversation_with_gemini(user_input_2, solar_excel, demand_data_file, system_params)
    print(json.dumps(response_2, indent=2))
    start_time = time.time()
    print("\n" + "="*80 + "\n--- Running Demonstration: Baseline (Heuristic, Price-Aware) Simulation ---")
    user_input_3 = "run a heuristic plan for 2024-05-06"
    response_3 = run_conversation_with_gemini(user_input_3, solar_excel, demand_data_file, system_params)
    print(json.dumps(response_3, indent=2))
    end_time= time.time()
    print(f"Time taken: {end_time - start_time} seconds")
