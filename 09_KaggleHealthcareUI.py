class KaggleHealthcareUI:
  """
  Gradio UI specifically designed for Kaggle healthcare dataset
  """

  def __init__(self, agent: KaggleHealthcareAgent):
    self.agent = agent

  def create_interface(self):
    """
    Create the Gradio interface
    """
    with gr.Blocks(title="Kaggle Healthcare Synthetic Data Generator", theme=gr.themes.Soft()) as interface:

        gr.Markdown("""
        # 🏥 Kaggle Healthcare Dataset - Synthetic Data Generator

        Generate privacy-compliant synthetic healthcare data based on learned patterns from the Kaggle healthcare dataset.

        ## 🔒 Privacy Guarantee
        ✅ 100% synthetic data • ✅ No real patient information • ✅ HIPAA compliant
        """)

        with gr.Tabs():
            # Main Generation Tab
            with gr.TabItem("🎯 Generate Synthetic Data"):
                with gr.Row():
                    with gr.Column(scale=1):
                        query_input = gr.Textbox(
                            label="Describe the Healthcare Data You Need",
                            placeholder="Example: Generate diabetes patients aged 50-70 with emergency admissions",
                            lines=3,
                            value="Generate patients with diabetes and hypertension"
                        )

                        num_records = gr.Slider(
                            label="Number of Records",
                            minimum=1,
                            maximum=50,
                            value=10,
                            step=1
                        )

                        generate_btn = gr.Button(
                            "🚀 Generate Synthetic Data",
                            variant="primary",
                            size="lg"
                        )

                        status_output = gr.Markdown(
                            value="Ready to generate synthetic data based on Kaggle healthcare patterns..."
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### Generated Synthetic Data")

                        data_display = gr.DataFrame(
                            label="Synthetic Healthcare Records",
                            interactive=False,
                            wrap=True
                        )

                        download_btn = gr.DownloadButton(
                            "📥 Download CSV File",
                            variant="secondary",
                            size="lg",
                            visible=False
                        )

                # Quick Examples
                gr.Markdown("### 💡 Example Queries")
                with gr.Row():
                    example_btns = [
                        gr.Button("👥 Diabetes Patients", size="sm"),
                        gr.Button("❤️ Cardiac Conditions", size="sm"),
                        gr.Button("🫁 Respiratory Issues", size="sm"),
                        gr.Button("🚑 Emergency Admissions", size="sm")
                    ]

            # # Dataset Insights Tab
            with gr.TabItem("📊 Dataset Insights"):
                insights_btn = gr.Button("🔍 Show Dataset Analysis", variant="primary")
                insights_output = gr.JSON(label="Dataset Analysis Results")

        # Event Handlers
        def handle_generation(query, num_records):
            try:
                result = self.agent.generate_synthetic_data(query, int(num_records))

                if result['success']:
                    # Parse CSV data
                    csv_data = result['synthetic_data']
                    from io import StringIO
                    df = pd.read_csv(StringIO(csv_data))

                    status_msg = f"""
                    ✅ **Successfully Generated {len(df)} Synthetic Records**

                    **Query:** {result['query']}
                    **Pattern Sources:** {result['pattern_sources']}
                    **Dataset Source:** Kaggle Healthcare Patterns
                    **Privacy Compliant:** ✅ Yes

                    ⚠️ **Note:** This is completely synthetic data generated from learned patterns.
                    """

                    # Create download file
                    import tempfile
                    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
                    temp_file.write(csv_data)
                    temp_file.close()

                    return (
                        df,
                        status_msg,
                        gr.DownloadButton("📥 Download CSV File", value=temp_file.name, visible=True)
                    )
                else:
                    error_msg = f"❌ Generation failed: {result.get('error', 'Unknown error')}"
                    return (None, error_msg, gr.DownloadButton(visible=False))

            except Exception as e:
                return (None, f"❌ Error: {str(e)}", gr.DownloadButton(visible=False))

        # def get_insights():
        #     try:
        #         insights = self.agent.get_dataset_insights()
        #         return insights
        #     except Exception as e:
        #         return {"error": str(e)}
        def get_insights():
          try:
            insights = self.agent.get_dataset_insights()

            # Additional safety check
            if 'error' in insights:
                return {
                    'status': 'error',
                    'message': insights['error'],
                    'timestamp': datetime.now().isoformat()
                }

            # Add timestamp for user reference
            insights['generated_at'] = datetime.now().isoformat()
            insights['data_source'] = 'Kaggle Healthcare Dataset'

            return insights

          except Exception as e:
              return {
                  'status': 'error',
                  'message': f'UI Error: {str(e)}',
                  'timestamp': datetime.now().isoformat(),
                  'suggestion': 'Try refreshing the system or check initialization'
              }

        # Wire up the event handler
        insights_btn.click(
            fn=get_insights,
            outputs=[insights_output]
        )

        # Example functions
        def set_diabetes_example():
            return "Generate patients with diabetes aged 45-75 with various admission types", 15

        def set_cardiac_example():
            return "Generate patients with hypertension and heart conditions requiring emergency care", 12

        def set_respiratory_example():
            return "Generate patients with asthma and respiratory conditions", 10

        def set_emergency_example():
            return "Generate emergency admission records with various critical conditions", 20

        # Wire up events
        generate_btn.click(
            fn=handle_generation,
            inputs=[query_input, num_records],
            outputs=[data_display, status_output, download_btn]
        )

        insights_btn.click(
            fn=get_insights,
            outputs=[insights_output]
        )

        # Example buttons
        example_btns[0].click(
            fn=set_diabetes_example,
            outputs=[query_input, num_records]
        )

        example_btns[1].click(
            fn=set_cardiac_example,
            outputs=[query_input, num_records]
        )

        example_btns[2].click(
            fn=set_respiratory_example,
            outputs=[query_input, num_records]
        )

        example_btns[3].click(
            fn=set_emergency_example,
            outputs=[query_input, num_records]
        )

    return interface

# ========== CELL 31 ==========
def setup_kaggle_system(dataset_path: str):
