if __name__ == '__main__':
    # --- 1. Evaluation Setup ---
    EVALUATION_DATASET = [
        # Tool: mpc_simulation
        {"query": "Plan energy for 2024-08-12.", "expected_tool": "mpc_simulation", "expected_params": {"date": "2024-08-12"}},
        {"query": "I need an optimized energy schedule for tomorrow.", "expected_tool": "mpc_simulation", "expected_params": {"date": "tomorrow"}},
        {"query": "pln the enrgy for 2024-08-12", "expected_tool": "mpc_simulation", "expected_params": {"date": "2024-08-12"}},
        {"query": "run mpc for yesterday", "expected_tool": "mpc_simulation", "expected_params": {"date": "yesterday"}},
        {"query": "Can you create an MPC plan for 2024-06-15?", "expected_tool": "mpc_simulation", "expected_params": {"date": "2024-06-15"}},
        {"query": "I need the energy optimization for the day after tomorrow.", "expected_tool": "mpc_simulation", "expected_params": {"date": "the day after tomorrow"}},

        # Tool: baseline_reactive_price_blind
        {"query": "What's the baseline for 2024-07-20?", "expected_tool": "baseline_reactive_price_blind", "expected_params": {"date": "2024-07-20"}},
        {"query": "generate a basic energy plan for today", "expected_tool": "baseline_reactive_price_blind", "expected_params": {"date": "today"}},
        {"query": "run a basic simulation for yesterday", "expected_tool": "baseline_reactive_price_blind", "expected_params": {"date": "yesterday"}},
        {"query": "show me the price-blind plan for tomorrow", "expected_tool": "baseline_reactive_price_blind", "expected_params": {"date": "tomorrow"}},

        # Tool: heuristic_price_aware_baseline
        {"query": "run a heuristic plan for today", "expected_tool": "heuristic_price_aware_baseline", "expected_params": {"date": "today"}},
        {"query": "what is the price-aware baseline for 2024-08-15?", "expected_tool": "heuristic_price_aware_baseline", "expected_params": {"date": "2024-08-15"}},
        {"query": "generate a rule-based energy plan for tomorrow", "expected_tool": "heuristic_price_aware_baseline", "expected_params": {"date": "tomorrow"}},


        # Tool: get_solar_forecast & plot_solar_forecast
        {"query": "Get solar forecast for tomorrow.", "expected_tool": "get_solar_forecast", "expected_params": {"date": "tomorrow"}},
        {"query": "Show me the PV numbers for 2024-09-10.", "expected_tool": "get_solar_forecast", "expected_params": {"date": "2024-09-10"}},
        {"query": "what was the solar output on 2024-02-28?", "expected_tool": "get_solar_forecast", "expected_params": {"date": "2024-02-28"}},
        {"query": "Plot the solar forecast for today.", "expected_tool": "plot_solar_forecast", "expected_params": {"date": "today"}},
        {"query": "Can you graph the sun power for yesterday?", "expected_tool": "plot_solar_forecast", "expected_params": {"date": "yesterday"}},
        {"query": "graph the pv power for 2024-07-01", "expected_tool": "plot_solar_forecast", "expected_params": {"date": "2024-07-01"}},

        # Tool: get_electricity_price & plot_electricity_price
        {"query": "What is the day-ahead price for 2024-10-20?", "expected_tool": "get_electricity_price", "expected_params": {"date": "2024-10-20", "market_type": "day-ahead"}},
        {"query": "real-time cost for today?", "expected_tool": "get_electricity_price", "expected_params": {"date": "today", "market_type": "real-time"}},
        {"query": "show me the day ahead market prices for 2024-12-25", "expected_tool": "get_electricity_price", "expected_params": {"date": "2024-12-25", "market_type": "day-ahead"}},
        {"query": "plot the real time market price for 2024-01-01", "expected_tool": "plot_electricity_price", "expected_params": {"date": "2024-01-01", "market_type": "real-time"}},
        {"query": "plot real-time electricity cost for tomorrow", "expected_tool": "plot_electricity_price", "expected_params": {"date": "tomorrow", "market_type": "real-time"}},

        # Tool: suggest_optimal_appliance_time_with_mpc
        {"query": "When should I run the Washing Machine?", "expected_tool": "suggest_optimal_appliance_time_with_mpc", "expected_params": {"appliance_name": "Washing Machine"}},
        {"query": "Best time for Dish Washer on 2024-11-01", "expected_tool": "suggest_optimal_appliance_time_with_mpc", "expected_params": {"appliance_name": "Dish Washer", "date": "2024-11-01"}},
        {"query": "what is the best time to use the geyser tomorrow", "expected_tool": "suggest_optimal_appliance_time_with_mpc", "expected_params": {"appliance_name": "Geyser (10 l)", "date": "tomorrow"}},
        {"query": "What's a good time to run the Iron today?", "expected_tool": "suggest_optimal_appliance_time_with_mpc", "expected_params": {"appliance_name": "Iron", "date": "today"}},

        # Out-of-Scope Queries
        {"query": "Hello, how are you?", "expected_tool": "N/A", "expected_params": {}},
        {"query": "What's the weather like?", "expected_tool": "N/A", "expected_params": {}},
        {"query": "Thank you!", "expected_tool": "N/A", "expected_params": {}},
        {"query": "is it going to rain?", "expected_tool": "N/A", "expected_params": {}},
        {"query": "How does the battery work?", "expected_tool": "N/A", "expected_params": {}},
        {"query": "Good morning", "expected_tool": "N/A", "expected_params": {}},
    ]

    def keyword_baseline_parser(query: str) -> Dict:
        """A simple keyword-based parser for baseline comparison."""
        q = query.lower()
        if "heuristic" in q or "rule-based" in q or "price-aware" in q: tool = "heuristic_price_aware_baseline"
        elif "baseline" in q or "basic" in q or "price-blind" in q: tool = "baseline_reactive_price_blind"
        elif "plan" in q or "mpc" in q or "schedule" in q or "optimization" in q: tool = "mpc_simulation"
        elif "plot" in q and ("solar" in q or "pv" in q or "sun" in q): tool = "plot_solar_forecast"
        elif "solar" in q or "pv" in q or "sun" in q: tool = "get_solar_forecast"
        elif "plot" in q and ("price" in q or "cost" in q): tool = "plot_electricity_price"
        elif "price" in q or "cost" in q: tool = "get_electricity_price"
        elif "when" in q or "best time" in q or "optimal" in q: tool = "suggest_optimal_appliance_time_with_mpc"
        else: return {"tool_name": "N/A", "parameters": {}}

        params = {}
        date_match = re.search(r'\d{4}-\d{2}-\d{2}', q) or re.search(r'tomorrow|today|yesterday', q)
        if date_match: params["date"] = date_match.group(0)
        return {"tool_name": tool, "parameters": params}

    def run_evaluation(dataset: List[Dict], eval_func):
        results, latencies = [], []
        print(f"\n--- Running Evaluation for: {eval_func.__name__} ---")
        for item in dataset:
            start = time.time()
            pred = eval_func(item["query"])
            latencies.append(time.time() - start)
            results.append({"expected_tool": item["expected_tool"], "predicted_tool": pred.get("tool_name", "N/A"),
                            "expected_params": item["expected_params"], "predicted_params": pred.get("parameters", {})})
        print("--- Evaluation Complete ---")
        return results, latencies

    def generate_performance_report(results: List[Dict], latencies: List[float], model_name: str):
        y_true = [r["expected_tool"] for r in results]
        y_pred = [r["predicted_tool"] for r in results]
        all_tools = sorted(list(set(y_true + y_pred)))
        tool_accuracy = np.mean([1 if yt == yp else 0 for yt, yp in zip(y_true, y_pred)]) * 100
        f1_scores = f1_score(y_true, y_pred, labels=all_tools, average=None, zero_division=0)
        per_tool_f1 = {tool: score for tool, score in zip(all_tools, f1_scores)}
        param_true_positives, param_predictions_count, param_ground_truths_count = 0, 0, 0
        for res in results:
            if res['expected_tool'] == res['predicted_tool'] and res['expected_tool'] != "N/A":
                expected_params, predicted_params = res.get('expected_params', {}), res.get('predicted_params', {})
                param_ground_truths_count += len(expected_params)
                param_predictions_count += len(predicted_params)
                for p_name, p_val in expected_params.items():
                    if p_name in predicted_params and predicted_params[p_name] == p_val:
                        param_true_positives += 1
        precision = param_true_positives / param_predictions_count if param_predictions_count > 0 else 0
        recall = param_true_positives / param_ground_truths_count if param_ground_truths_count > 0 else 0
        param_f1 = (2 * precision * recall) / (precision + recall) * 100 if (precision + recall) > 0 else 0

        print("\n" + "="*50 + f"\n  Performance Report for: {model_name}\n" + "="*50)
        print(f"Overall Tool-Name Accuracy: {tool_accuracy:.2f}%")
        print(f"Parameter Extraction F1-Score: {param_f1:.2f}%")
        print(f"Average Latency per Query: {np.mean(latencies):.4f} seconds")
        print("\n--- Per-Tool F1 Scores ---")
        for tool, score in per_tool_f1.items(): print(f"- {tool}: {score:.2f}")
        print("\n" + "="*50)

    # --- Run Evaluations for all configured models ---
    print("\n" + "="*80 + "\n" + " " * 25 + "STARTING MODEL EVALUATION" + "\n" + "="*80)

    # 1. Gemini Model
    gemini_results, gemini_latencies = run_evaluation(EVALUATION_DATASET, call_gemini_for_tool)
    generate_performance_report(gemini_results, gemini_latencies, "Gemini 1.5 Flash")

    # 2. Keyword Baseline
    baseline_results, baseline_latencies = run_evaluation(EVALUATION_DATASET, keyword_baseline_parser)
    generate_performance_report(baseline_results, baseline_latencies, "Keyword Baseline")

    # 3. Ollama Models Evaluation
    ollama_llama_results = None
    ollama_phi_results = None
    ollama_gemma_results = None

    if ollama_client:
        print("\nOllama client is connected. Proceeding with Ollama evaluations.")

        # Llama 3
        print("\n--- Evaluating: Ollama Llama 3 ---")
        ollama_llama_eval_func = partial(call_ollama_for_tool, model_name='llama3')
        ollama_llama_eval_func.__name__ = "call_ollama_for_tool_llama3"
        ollama_llama_results, ollama_llama_latencies = run_evaluation(EVALUATION_DATASET, ollama_llama_eval_func)
        generate_performance_report(ollama_llama_results, ollama_llama_latencies, "Ollama Llama 3")

        # Phi-3 Mini
        print("\n--- Evaluating: Ollama Phi-3 Mini ---")
        ollama_phi_eval_func = partial(call_ollama_for_tool, model_name='phi3:mini')
        ollama_phi_eval_func.__name__ = "call_ollama_for_tool_phi3"
        ollama_phi_results, ollama_phi_latencies = run_evaluation(EVALUATION_DATASET, ollama_phi_eval_func)
        generate_performance_report(ollama_phi_results, ollama_phi_latencies, "Ollama Phi-3 Mini")

        # Gemma 7B
        print("\n--- Evaluating: Ollama Gemma 7B ---")
        ollama_gemma_eval_func = partial(call_ollama_for_tool, model_name='gemma:7b')
        ollama_gemma_eval_func.__name__ = "call_ollama_for_tool_gemma"
        ollama_gemma_results, ollama_gemma_latencies = run_evaluation(EVALUATION_DATASET, ollama_gemma_eval_func)
        generate_performance_report(ollama_gemma_results, ollama_gemma_latencies, "Ollama Gemma 7B")


    else:
        print("\n" + "="*80)
        print("SKIPPING OLLAMA EVALUATION: Ollama client not available or not connected.")
        print("Please ensure the Ollama server is running and the setup cell was completed.")
        print("="*80)


    # Generate the summary files with all results
    generate_summary_markdown_file(EVALUATION_DATASET, gemini_results, baseline_results, ollama_llama_results, ollama_phi_results, ollama_gemma_results)
    generate_summary_excel_file(EVALUATION_DATASET, gemini_results, baseline_results, ollama_llama_results, ollama_phi_results, ollama_gemma_results)


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


