def setup_kaggle_system(dataset_path: str):
  """
  Setup the Kaggle healthcare system with existing index check
  """
  print("🚀 Setting up Kaggle Healthcare Synthetic Data System...")

  try:
    # Verify dataset exists
    if not os.path.exists(dataset_path):
      raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Initialize agent
    agent = KaggleHealthcareAgent(dataset_path)

    # Check if system is already initialized
    if check_existing_system(agent):
      print("✅ System already initialized! Skipping setup...")
      # Just initialize the data generator without reprocessing
      agent.analyzer.load_and_analyze()  # Quick analysis load
      agent.data_generator = KaggleSyntheticDataGenerator(agent.vector_manager, agent.analyzer)
      agent.initialized = True

      # Get basic stats for display
      df = pd.read_csv(dataset_path)
      return {
          'agent': agent,
          'dataset_records': len(df),
          'patterns_learned': 'existing',
          'distributions_analyzed': 'existing',
          'relationships_discovered': 'existing',
          'setup_type': 'quick_load'
      }

    # If not initialized, run full setup
    print("🔄 Running full system initialization...")
    result = agent.initialize_system()

    if result['success']:
      print(f"✅ System Setup Complete!")
      print(f"📊 Dataset Records: {result['dataset_records']:,}")
      print(f"🧠 Patterns Learned: {result['patterns_learned']}")
      print(f"📈 Distributions Analyzed: {result['distributions_analyzed']}")
      print(f"🔗 Relationships Discovered: {result['relationships_discovered']}")
      result['agent'] = agent
      result['setup_type'] = 'full_setup'
      return result
    else:
      print(f"❌ Setup failed: {result['error']}")
      return None

  except Exception as e:
    print(f"❌ Error setting up system: {str(e)}")
    return None


def check_existing_system(agent: KaggleHealthcareAgent) -> bool:
  """
  Check if the Pinecone index already exists and has data
  """
  try:
    print("🔍 Checking for existing system...")

    # Check if Pinecone index exists
    existing_indexes = [index.name for index in agent.vector_manager.pc.list_indexes()]

    if agent.vector_manager.index_name not in existing_indexes:
      print("📝 Pinecone index not found - need full setup")
      return False

    # Check if index has data
    stats = agent.vector_manager.index.describe_index_stats()
    total_vectors = stats.get('total_vector_count', 0)

    if total_vectors == 0:
      print("📝 Pinecone index empty - need full setup")
      return False

    print(f"✅ Found existing system with {total_vectors} vectors")
    return True

  except Exception as e:
    print(f"⚠️ Error checking existing system: {e}")
    return False


def setup_kaggle_system_fast(dataset_path: str):
  """
  OPTIMIZED: Faster setup with existing system check
  """
  print("⚡ Setting up Kaggle Healthcare System (FAST MODE)...")

  try:
    # Step 1: Quick environment check
    assert os.getenv('ANTHROPIC_API_KEY'), "❌ Anthropic API key not found"
    assert os.getenv('VOYAGE_API_KEY'), "❌ Voyage AI API key not found"
    assert os.getenv('PINECONE_API_KEY'), "❌ Pinecone API key not found"
    print("✅ Environment verified")

    # Verify dataset exists
    if not os.path.exists(dataset_path):
      raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Step 2: Initialize agent (lightweight)
    agent = KaggleHealthcareAgent(dataset_path)

    # Step 3: Check if already setup
    if check_existing_system(agent):
      print("⚡ EXISTING SYSTEM FOUND - QUICK LOAD!")

      # Quick setup without reprocessing
      patterns, distributions, relationships = agent.analyzer.load_and_analyze()
      agent.data_generator = KaggleSyntheticDataGenerator(agent.vector_manager, agent.analyzer)
      agent.initialized = True

      df = pd.read_csv(dataset_path)

      print(f"✅ Quick load complete!")
      print(f"📊 Dataset Records: {len(df):,}")
      print(f"🧠 Using existing vector embeddings")
      print(f"🚀 Ready for synthetic data generation!")

      return agent, len(df)

    # Step 4: Full setup if needed
    print("🔄 No existing system found - running full setup...")
    result = agent.initialize_system()

    if result['success']:
      print(f"✅ Full setup complete!")
      print(f"📊 Dataset Records: {result['dataset_records']:,}")
      print(f"🧠 Patterns Learned: {result['patterns_learned']}")
      return agent, result['dataset_records']
    else:
      print(f"❌ Setup failed: {result['error']}")
      return None, 0

  except Exception as e:
    print(f"❌ Fast setup failed: {e}")
    return None, 0

# ========== CELL 32 ==========
# Function to run the enhanced analysis and update the system for optimisation
def upgrade_to_medically_accurate_system(dataset_path: str, agent):
  """
  Upgrade the system to use medically accurate relationships
  """
  print("🔬 UPGRADING TO MEDICALLY ACCURATE SYSTEM")
  print("="*60)

  # global healthcare_agent
  # Get dataset path from existing agent
  if not dataset_path:
    dataset_path = agent.dataset_path

  # Step 1: Create enhanced analyzer
  print("Step 1: Creating enhanced dataset analyzer...")
  enhanced_analyzer = EnhancedKaggleDatasetAnalyzer(dataset_path)

  # Step 2: Perform deep analysis
  print("Step 2: Performing deep medical relationship analysis...")
  analysis_summary = enhanced_analyzer.load_and_deep_analyze()

  # Step 3: Print medical accuracy report
  print("Step 3: Generating medical accuracy report...")
  issues_found = enhanced_analyzer.print_medical_accuracy_report()

  # Step 3: PRESERVE ORIGINAL DISTRIBUTIONS, PATTERNS, RELATIONSHIPS
  print("Step 3: Preserving original analyzer data...")
  if hasattr(agent.analyzer, 'distributions'):
    enhanced_analyzer.distributions = agent.analyzer.distributions
  if hasattr(agent.analyzer, 'patterns'):
    enhanced_analyzer.patterns = agent.analyzer.patterns
  if hasattr(agent.analyzer, 'relationships'):
    enhanced_analyzer.relationships = agent.analyzer.relationships

  # Step 5: Update the global healthcare agent
  print("Step 5: Updating system with enhanced analyzer...")


  # Replace the analyzer in the existing system
  healthcare_agent.analyzer = enhanced_analyzer
  healthcare_agent.data_generator = MedicallyAccurateDataGenerator(
      healthcare_agent.vector_manager,
      enhanced_analyzer
  )

  print(f"\n✅ SYSTEM UPGRADE COMPLETE!")
  print(f"🔍 Medical Accuracy Issues Identified: {issues_found}")
  print(f"📊 Realistic Combinations Discovered: {analysis_summary['total_realistic_combinations']}")
  print(f"💊 Condition-Medication Mappings: {analysis_summary['total_conditions']}")

  return enhanced_analyzer, issues_found

dataset_path = "/content/uom-msc-cs1/tmp/healthcare_dataset.csv"

# Run the upgrade
print("🚀 STARTING MEDICAL ACCURACY UPGRADE...")
enhanced_analyzer, issues_found = upgrade_to_medically_accurate_system(dataset_path, healthcare_agent)

print(f"\n🎯 TESTING IMPROVED ACCURACY...")
print("Now testing the same query that gave incorrect results...")

# Test the problematic query
test_query = "Generate patients with asthma requiring medication treatment"
result = healthcare_agent.generate_synthetic_data(test_query, 5)

if result['success']:
    print(f"✅ Generated data for: {test_query}")
    print("📄 Sample of generated data:")
    print(result['synthetic_data'][:500] + "...")
else:
    print(f"❌ Generation failed: {result.get('error')}")

# ========== CELL 33 ==========
# # Main execution functions
# def setup_kaggle_system(dataset_path: str):
#     """
#     Setup the Kaggle healthcare system
#     """
#     print("🚀 Setting up Kaggle Healthcare Synthetic Data System...")

#     try:
#         # Verify dataset exists
#         if not os.path.exists(dataset_path):
#             raise FileNotFoundError(f"Dataset not found: {dataset_path}")

#         # Initialize agent
#         agent = KaggleHealthcareAgent(dataset_path)

#         # Initialize system (this will take time for analysis and vector creation)
#         result = agent.initialize_system()

#         if result['success']:
#             print(f"✅ System Setup Complete!")
#             print(f"📊 Dataset Records: {result['dataset_records']:,}")
#             print(f"🧠 Patterns Learned: {result['patterns_learned']}")
#             print(f"📈 Distributions Analyzed: {result['distributions_analyzed']}")
#             print(f"🔗 Relationships Discovered: {result['relationships_discovered']}")
#             return agent
#         else:
#             print(f"❌ Setup failed: {result['error']}")
#             return None

#     except Exception as e:
#         print(f"❌ Error setting up system: {str(e)}")
#         return None

def launch_kaggle_ui(agent: KaggleHealthcareAgent):
  """
  Launch the Kaggle healthcare UI
  """
  if agent is None:
    print("❌ No agent provided. Setup the system first!")
    return

  try:
    print("🌐 Launching Kaggle Healthcare UI...")

    ui = KaggleHealthcareUI(agent)
    interface = ui.create_interface()

    print("="*60)
    print("🏥 KAGGLE HEALTHCARE SYNTHETIC DATA GENERATOR")
    print("="*60)
    print("🎯 Focused on Kaggle healthcare dataset patterns")
    print("🔒 Privacy-first synthetic data generation")
    print("🧠 AI-powered pattern learning and replication")
    print("📊 Real distribution-based data generation")
    print("="*60)

    interface.launch(
        share=True,
        debug=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )

  except Exception as e:
    print(f"❌ Error launching UI: {str(e)}")

# ========== CELL 34 ==========
# Update the main execution section
if __name__ == "__main__":
  print("""
    🏥 Kaggle Healthcare Dataset - Synthetic Data Generator
    ====================================================

    🎯 FOCUSED APPROACH:
    - Exclusively designed for Kaggle healthcare dataset
    - Deep pattern learning from your specific data
    - Distribution-aware synthetic data generation
    - Relationship-preserving data synthesis

    🔒 PRIVACY PROTECTION:
    - Source data is anonymized during processing
    - Users only access synthetic outputs
    - No real patient data exposure
    - HIPAA compliant synthetic generation

    📊 PATTERN LEARNING:
    - Age-condition relationships
    - Medication-condition associations
    - Billing pattern analysis
    - Admission type distributions
    - Test result patterns
    """)

  # Set your dataset path here
  DATASET_PATH = "/content/uom-msc-cs1/tmp/healthcare_dataset.csv"  # Update this path

  print(f"\n🔍 Looking for dataset at: {DATASET_PATH}")

  if os.path.exists(DATASET_PATH):
    print("✅ Dataset found! Setting up system...")

    # Use FAST setup with existing system check
    healthcare_agent, dataset_count = setup_kaggle_system_fast(DATASET_PATH)

    if healthcare_agent:
      print("\n🚀 System ready! Launching UI...")
      enhanced_analyzer, issues_found = upgrade_to_medically_accurate_system(DATASET_PATH, healthcare_agent)
      # Launch UI
      # launch_kaggle_ui(healthcare_agent)
    else:
      print("❌ System setup failed")
  else:
    print(f"❌ Dataset not found at {DATASET_PATH}")
    print("Please update DATASET_PATH variable with the correct path to your healthcare_dataset.csv file")
    print("\nAlternative paths to try:")
    print("- /content/healthcare_dataset.csv")
    print("- /content/tmp/healthcare_dataset.csv")
    print("- /kaggle/input/healthcare-dataset/healthcare_dataset.csv")
    print("- ./healthcare_dataset.csv")

# ========== CELL 35 ==========
# To launch the UI
launch_kaggle_ui(healthcare_agent)

# ========== CELL 36 ==========
def test_kaggle_similarity_scores():
    """
    Test enhanced similarity scores for Kaggle healthcare dataset
    """

    test_categories = {
        "Diabetes Management": [
            "Generate patients with diabetes aged 50-70",
            "Generate diabetic patients with insulin medication",
            "Generate diabetes patients with abnormal test results",
            "Generate Type 2 diabetes patients with emergency admissions"
        ],

        "Hypertension/Cardiovascular": [
            "Generate patients with hypertension and heart conditions",
            "Generate cardiac patients with lisinopril medication",
            "Generate hypertension patients aged 60-80",
            "Generate cardiovascular patients with elective admissions"
        ],

        "Cancer Treatment": [
            "Generate cancer patients with chemotherapy treatment",
            "Generate oncology patients with abnormal test results",
            "Generate cancer patients with high billing amounts",
            "Generate malignancy patients requiring urgent care"
        ],

        "Respiratory Conditions": [
            "Generate asthma patients with inhaler medications",
            "Generate respiratory patients with normal test results",
            "Generate COPD patients with emergency admissions",
            "Generate pulmonary patients aged 40-65"
        ],

        "Emergency Admissions": [
            "Generate emergency admission patients with critical conditions",
            "Generate urgent care patients with various medical conditions",
            "Generate emergency department visits with abnormal results",
            "Generate critical patients requiring immediate intervention"
        ],

        "Elective Procedures": [
            "Generate elective admission patients for routine care",
            "Generate scheduled patients with normal test results",
            "Generate elective surgery patients with moderate billing",
            "Generate planned admission patients with chronic conditions"
        ],

        "Age-Specific Patterns": [
            "Generate elderly patients aged 70-85 with multiple conditions",
            "Generate young adults aged 25-40 with acute conditions",
            "Generate middle-aged patients 45-60 with chronic diseases",
            "Generate pediatric equivalent adult patients with developmental issues"
        ],

        "Medication Patterns": [
            "Generate patients prescribed aspirin for cardiovascular protection",
            "Generate patients on ibuprofen for pain management",
            "Generate patients with paracetamol for fever control",
            "Generate patients on multiple medications for chronic conditions"
        ],

        "Insurance and Billing": [
            "Generate patients with high billing amounts over $50000",
            "Generate patients with Medicare insurance coverage",
            "Generate patients with Blue Cross insurance plans",
            "Generate patients with moderate billing between $10000-30000"
        ],

        "Test Results Patterns": [
            "Generate patients with abnormal test results requiring follow-up",
            "Generate patients with normal test results for routine care",
            "Generate patients with inconclusive test results needing repeat testing",
            "Generate patients with test results matching their medical conditions"
        ],

        "Hospital and Room Assignment": [
            "Generate patients in rooms 200-299 on second floor",
            "Generate patients in private rooms with premium care",
            "Generate patients in specific hospitals for specialized care",
            "Generate patients with extended stays in hospital rooms"
        ],

        "Gender-Specific Conditions": [
            "Generate female patients with gender-specific conditions",
            "Generate male patients with cardiovascular risk factors",
            "Generate patients with gender-balanced chronic conditions",
            "Generate elderly women with osteoporosis and related conditions"
        ],

        "Blood Type Correlations": [
            "Generate patients with O+ blood type and common conditions",
            "Generate patients with rare blood types AB- requiring special care",
            "Generate patients with A+ blood type and diabetes",
            "Generate patients with blood type considerations for treatment"
        ],

        "Comprehensive Patient Records": [
            "Generate complete patient records with all medical details",
            "Generate comprehensive healthcare records for chronic patients",
            "Generate detailed patient profiles with multiple data points",
            "Generate full medical records for longitudinal care tracking"
        ],

        "Seasonal and Temporal Patterns": [
            "Generate patients admitted in winter months with respiratory issues",
            "Generate patients with seasonal condition exacerbations",
            "Generate patients with admission dates in 2023-2024 period",
            "Generate patients with realistic discharge timing patterns"
        ]
    }

    print("🧪 TESTING KAGGLE HEALTHCARE SIMILARITY SCORES")
    print("="*85)
    print(f"{'Category':<25} {'Query':<40} {'Score':<8} {'Quality'}")
    print("="*85)

    category_results = {}
    total_tests = 0
    successful_tests = 0

    for category, queries in test_categories.items():
        category_scores = []
        print(f"\n🔍 Testing Category: {category}")
        print("-" * 85)

        for query in queries:
            total_tests += 1
            try:
                # Test similarity search using the Kaggle-specific search method
                patterns = healthcare_agent.vector_manager.search_similar_patterns(query, top_k=5)

                if patterns:
                    # Get best score
                    best_score = patterns[0].get('score', 0)
                    category_scores.append(best_score)
                    successful_tests += 1

                    # Determine quality based on Kaggle dataset expectations
                    if best_score >= 0.80:
                        quality = "🏆 OUTSTANDING"
                        quality_color = "🟩"
                    elif best_score >= 0.70:
                        quality = "✅ EXCELLENT"
                        quality_color = "🟩"
                    elif best_score >= 0.60:
                        quality = "✅ GOOD"
                        quality_color = "🟨"
                    elif best_score >= 0.50:
                        quality = "⚠️ MODERATE"
                        quality_color = "🟨"
                    elif best_score >= 0.40:
                        quality = "⚠️ FAIR"
                        quality_color = "🟧"
                    else:
                        quality = "❌ POOR"
                        quality_color = "🟥"

                    # Print result with better formatting
                    short_query = query[:35] + "..." if len(query) > 35 else query
                    print(f"{category[:24]:<25} {short_query:<40} {best_score:.3f}    {quality_color} {quality}")

                    # Show top pattern metadata for context
                    if 'metadata' in patterns[0]:
                        metadata = patterns[0]['metadata']
                        condition = metadata.get('medical_condition', 'Unknown')
                        age_group = metadata.get('age_group', 'Unknown')
                        print(f"{'':>25} {'↳ Match: ' + condition + ' (' + age_group + ')':<40}")

                else:
                    print(f"{category[:24]:<25} {query[:35]:<40} {'N/A':<8} ❌ NO RESULTS")

            except Exception as e:
                print(f"{category[:24]:<25} {query[:35]:<40} {'ERROR':<8} ❌ {str(e)[:20]}")
                continue

        # Calculate category statistics
        if category_scores:
            avg_score = sum(category_scores) / len(category_scores)
            min_score = min(category_scores)
            max_score = max(category_scores)
            std_dev = (sum([(x - avg_score) ** 2 for x in category_scores]) / len(category_scores)) ** 0.5

            category_results[category] = {
                'avg_score': avg_score,
                'min_score': min_score,
                'max_score': max_score,
                'std_dev': std_dev,
                'scores': category_scores,
                'count': len(category_scores)
            }

    # Detailed Summary
    print(f"\n📊 DETAILED CATEGORY ANALYSIS")
    print("="*80)
    print(f"{'Category':<25} {'Avg':<6} {'Min':<6} {'Max':<6} {'StdDev':<7} {'Status'}")
    print("="*80)

    for category, results in category_results.items():
        avg_score = results['avg_score']
        min_score = results['min_score']
        max_score = results['max_score']
        std_dev = results['std_dev']

        if avg_score >= 0.75:
            status = "🏆 OUTSTANDING"
        elif avg_score >= 0.65:
            status = "✅ EXCELLENT"
        elif avg_score >= 0.55:
            status = "✅ GOOD"
        elif avg_score >= 0.45:
            status = "⚠️ MODERATE"
        else:
            status = "❌ NEEDS WORK"

        print(f"{category[:24]:<25} {avg_score:.3f}  {min_score:.3f}  {max_score:.3f}  {std_dev:.3f}   {status}")

    # Overall Performance Analysis
    print(f"\n🎯 OVERALL SYSTEM PERFORMANCE")
    print("="*60)

    all_scores = []
    for results in category_results.values():
        all_scores.extend(results['scores'])

    if all_scores:
        overall_avg = sum(all_scores) / len(all_scores)
        overall_min = min(all_scores)
        overall_max = max(all_scores)
        overall_std = (sum([(x - overall_avg) ** 2 for x in all_scores]) / len(all_scores)) ** 0.5

        print(f"📈 Average Similarity Score: {overall_avg:.3f}")
        print(f"📊 Score Range: {overall_min:.3f} - {overall_max:.3f}")
        print(f"📏 Standard Deviation: {overall_std:.3f}")
        print(f"✅ Successful Tests: {successful_tests}/{total_tests} ({successful_tests/total_tests*100:.1f}%)")

        # Performance bands
        excellent_count = len([s for s in all_scores if s >= 0.70])
        good_count = len([s for s in all_scores if 0.60 <= s < 0.70])
        moderate_count = len([s for s in all_scores if 0.50 <= s < 0.60])
        poor_count = len([s for s in all_scores if s < 0.50])

        print(f"\n📊 SCORE DISTRIBUTION:")
        print(f"🏆 Outstanding (≥0.70): {excellent_count} ({excellent_count/len(all_scores)*100:.1f}%)")
        print(f"✅ Good (0.60-0.69): {good_count} ({good_count/len(all_scores)*100:.1f}%)")
        print(f"⚠️ Moderate (0.50-0.59): {moderate_count} ({moderate_count/len(all_scores)*100:.1f}%)")
        print(f"❌ Poor (<0.50): {poor_count} ({poor_count/len(all_scores)*100:.1f}%)")

        # Overall assessment
        if overall_avg >= 0.75:
            print(f"\n🎉 SYSTEM STATUS: OUTSTANDING!")
            print("Your Kaggle healthcare system is performing exceptionally well!")
        elif overall_avg >= 0.65:
            print(f"\n✅ SYSTEM STATUS: EXCELLENT!")
            print("Strong similarity performance across medical categories!")
        elif overall_avg >= 0.55:
            print(f"\n👍 SYSTEM STATUS: GOOD!")
            print("Solid performance with room for minor improvements!")
        elif overall_avg >= 0.45:
            print(f"\n⚠️ SYSTEM STATUS: MODERATE")
            print("Acceptable performance but consider optimization!")
        else:
            print(f"\n🔧 SYSTEM STATUS: NEEDS IMPROVEMENT")
            print("Consider retraining or adjusting parameters!")

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if overall_avg < 0.60:
            print("• Consider increasing chunk_size in text splitter")
            print("• Try different embedding models")
            print("• Increase the number of similar patterns retrieved")

        if overall_std > 0.15:
            print("• High variation in scores - consider more consistent preprocessing")
            print("• Review query enhancement strategies")

        print("• Focus improvement on categories with lowest scores")
        print("• Consider domain-specific fine-tuning for medical terms")

    return category_results

# ========== CELL 37 ==========
# Function to run quick similarity test for specific queries
def quick_similarity_test(queries: list):
    """
    Quick test for specific queries
    """
    print("🚀 QUICK SIMILARITY TEST")
    print("="*50)

    for i, query in enumerate(queries, 1):
        try:
            patterns = healthcare_agent.vector_manager.search_similar_patterns(query, top_k=3)
            if patterns:
                score = patterns[0].get('score', 0)
                print(f"{i}. Query: {query}")
                print(f"   Score: {score:.3f} {'✅' if score >= 0.60 else '⚠️' if score >= 0.50 else '❌'}")
                if 'metadata' in patterns[0]:
                    condition = patterns[0]['metadata'].get('medical_condition', 'Unknown')
                    print(f"   Match: {condition}")
                print()
            else:
                print(f"{i}. Query: {query}")
                print(f"   Score: No results ❌")
                print()
        except Exception as e:
            print(f"{i}. Query: {query}")
            print(f"   Error: {str(e)} ❌")
            print()

# Run the comprehensive test
print("🧪 Starting Comprehensive Similarity Score Analysis...")
print("This will test the system's ability to find relevant patterns for various medical queries.")
print()

# Run the main test
category_results = test_kaggle_similarity_scores()

# Example of quick test for specific queries
print(f"\n" + "="*60)
print("🎯 QUICK TEST - Top Queries for Your Dataset")
print("="*60)

quick_test_queries = [
    "Generate diabetes patients aged 55-70 with insulin treatment",
    "Generate emergency admissions with abnormal test results",
    "Generate patients with hypertension and cardiac medications",
    "Generate cancer patients with high billing amounts",
    "Generate comprehensive patient records with multiple conditions"
]

quick_similarity_test(quick_test_queries)

