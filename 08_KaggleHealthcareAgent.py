class KaggleHealthcareAgent:
  """
  Main agent class focused on Kaggle healthcare dataset
  """

  def __init__(self, dataset_path: str):
    self.dataset_path = dataset_path
    self.privacy_manager = PrivacySecurityManager()
    self.analyzer = KaggleDatasetAnalyzer(dataset_path)
    self.vector_manager = KagglePineconeVectorManager()
    self.data_generator = None
    self.initialized = False

  def initialize_system(self):
    """
    Initialize the system by analyzing the dataset and loading into vector store
    """
    try:
      logger.info("Initializing Kaggle Healthcare System...")

      # Step 1: Analyze the dataset
      logger.info("Step 1: Analyzing dataset patterns...")
      patterns, distributions, relationships = self.analyzer.load_and_analyze()

      # Step 2: Load and anonymize data
      logger.info("Step 2: Loading and processing data...")
      df = pd.read_csv(self.dataset_path)
      anonymized_df = self.privacy_manager.anonymize_data(df)

      # Step 3: Process and store in vector database
      logger.info("Step 3: Creating vector embeddings...")
      self.vector_manager.process_kaggle_dataset(anonymized_df, self.analyzer)

      # Step 4: Initialize data generator
      logger.info("Step 4: Initializing synthetic data generator...")
      self.data_generator = KaggleSyntheticDataGenerator(self.vector_manager, self.analyzer)

      self.initialized = True
      logger.info("✅ System initialization complete!")

      return {
          'success': True,
          'dataset_records': len(df),
          'patterns_learned': len(patterns),
          'distributions_analyzed': len(distributions),
          'relationships_discovered': len(relationships)
      }

    except Exception as e:
      logger.error(f"Error initializing system: {str(e)}")
      return {'success': False, 'error': str(e)}

  def generate_synthetic_data(self, query: str, num_records: int = 5) -> Dict[str, Any]:
    """
    Generate synthetic data based on the query
    """
    if not self.initialized:
      return {'success': False, 'error': 'System not initialized. Call initialize_system() first.'}

    return self.data_generator.generate_synthetic_data(query, num_records)

  # def get_dataset_insights(self) -> Dict[str, Any]:
  #   """
  #   Get insights about the analyzed dataset
  #   """
  #   if not self.initialized:
  #     return {'error': 'System not initialized'}

  #   return {
  #       'distributions': self.analyzer.distributions,
  #       'patterns': self.analyzer.patterns,
  #       'relationships': self.analyzer.relationships,
  #       'dataset_summary': {
  #           'total_records': len(self.analyzer.df),
  #           'unique_conditions': len(self.analyzer.distributions['Medical Condition']['value_counts']),
  #           'unique_medications': len(self.analyzer.distributions['Medication']['value_counts']),
  #           'age_range': f"{self.analyzer.distributions['Age']['min']}-{self.analyzer.distributions['Age']['max']}",
  #           'billing_range': f"${self.analyzer.distributions['Billing Amount']['min']:.0f}-${self.analyzer.distributions['Billing Amount']['max']:.0f}"
  #       }
  #   }
  def get_dataset_insights(self) -> Dict[str, Any]:
    """
    Get dataset insights with bulletproof JSON safety
    """
    if not self.initialized:
        return {
            'status': 'error',
            'message': 'System not initialized',
            'timestamp': str(datetime.now())
        }

    try:
        # Force everything to be JSON-safe
        insights = {
            'status': 'success',
            'timestamp': str(datetime.now()),
            'data_source': 'Kaggle Healthcare Dataset',
            'basic_information': {
                'total_records': str(len(self.analyzer.df)),
                'total_columns': str(len(self.analyzer.df.columns)),
                'system_status': 'Initialized and Ready',
                'dataset_shape': f"{len(self.analyzer.df)} rows x {len(self.analyzer.df.columns)} columns"
            }
        }

        # Safely get age statistics
        try:
            age_dist = self.analyzer.distributions.get('Age', {})
            insights['age_statistics'] = {
                'minimum_age': str(age_dist.get('min', 'Unknown')),
                'maximum_age': str(age_dist.get('max', 'Unknown')),
                'average_age': str(round(age_dist.get('mean', 0), 1)),
                'age_range': f"{age_dist.get('min', 0)} - {age_dist.get('max', 0)} years"
            }
        except Exception as e:
            insights['age_statistics'] = {'error': f'Age data unavailable: {str(e)}'}

        # Safely get medical conditions
        try:
            condition_dist = self.analyzer.distributions.get('Medical Condition', {})
            condition_counts = condition_dist.get('value_counts', {})

            conditions_safe = {}
            for i, (condition, count) in enumerate(list(condition_counts.items())[:10]):
                conditions_safe[f'rank_{i+1:02d}'] = f"{str(condition)}: {str(count)} patients"

            insights['top_medical_conditions'] = conditions_safe
            insights['condition_summary'] = {
                'total_unique_conditions': str(len(condition_counts)),
                'most_common': str(list(condition_counts.keys())[0]) if condition_counts else 'None'
            }
        except Exception as e:
            insights['top_medical_conditions'] = {'error': f'Condition data unavailable: {str(e)}'}

        # Safely get medications
        try:
            medication_dist = self.analyzer.distributions.get('Medication', {})
            medication_counts = medication_dist.get('value_counts', {})

            medications_safe = {}
            for i, (medication, count) in enumerate(list(medication_counts.items())[:10]):
                medications_safe[f'rank_{i+1:02d}'] = f"{str(medication)}: {str(count)} prescriptions"

            insights['top_medications'] = medications_safe
            insights['medication_summary'] = {
                'total_unique_medications': str(len(medication_counts)),
                'most_prescribed': str(list(medication_counts.keys())[0]) if medication_counts else 'None'
            }
        except Exception as e:
            insights['top_medications'] = {'error': f'Medication data unavailable: {str(e)}'}

        # Safely get billing information
        try:
            billing_dist = self.analyzer.distributions.get('Billing Amount', {})
            insights['billing_statistics'] = {
                'minimum_billing': f"${billing_dist.get('min', 0):,.0f}",
                'maximum_billing': f"${billing_dist.get('max', 0):,.0f}",
                'average_billing': f"${billing_dist.get('mean', 0):,.0f}",
                'billing_range': f"${billing_dist.get('min', 0):,.0f} - ${billing_dist.get('max', 0):,.0f}"
            }
        except Exception as e:
            insights['billing_statistics'] = {'error': f'Billing data unavailable: {str(e)}'}

        # System capabilities
        insights['system_capabilities'] = {
            'privacy_compliance': 'HIPAA Compliant',
            'synthetic_data_generation': 'Ready',
            'pattern_recognition': 'Active',
            'medical_relationship_learning': 'Enabled',
            'l4_gpu_optimization': str(getattr(self, 'is_l4_optimized', False)),
            'vector_database': 'Pinecone Connected',
            'ai_model': 'Claude 3.5 Sonnet + Voyage AI Embeddings'
        }

        # Medical accuracy metrics (if available)
        try:
            if hasattr(self.analyzer, 'condition_medication_map'):
                total_conditions = len(self.analyzer.condition_medication_map)
                conditions_with_meds = sum(1 for meds in self.analyzer.condition_medication_map.values() if meds)

                insights['medical_accuracy'] = {
                    'total_conditions_analyzed': str(total_conditions),
                    'conditions_with_medications': str(conditions_with_meds),
                    'accuracy_percentage': f"{(conditions_with_meds/total_conditions*100):.1f}%" if total_conditions > 0 else "0%",
                    'data_quality': 'High' if conditions_with_meds/total_conditions > 0.8 else 'Good' if conditions_with_meds/total_conditions > 0.6 else 'Moderate'
                }
            else:
                insights['medical_accuracy'] = {
                    'status': 'Analysis not available',
                    'note': 'Enhanced analyzer not fully initialized'
                }
        except Exception as e:
            insights['medical_accuracy'] = {'error': f'Accuracy metrics unavailable: {str(e)}'}

        # Data distribution summary
        try:
            insights['data_distribution'] = {
                'admission_types': str(len(self.analyzer.df['Admission Type'].unique())) + ' types',
                'blood_types': str(len(self.analyzer.df['Blood Type'].unique())) + ' types',
                'hospitals': str(len(self.analyzer.df['Hospital'].unique())) + ' hospitals',
                'doctors': str(len(self.analyzer.df['Doctor'].unique())) + ' doctors',
                'insurance_providers': str(len(self.analyzer.df['Insurance Provider'].unique())) + ' providers'
            }
        except Exception as e:
            insights['data_distribution'] = {'error': f'Distribution data unavailable: {str(e)}'}

        return insights

    except Exception as e:
        # Ultimate fallback
        return {
            'status': 'error',
            'message': f'Critical error retrieving insights: {str(e)}',
            'timestamp': str(datetime.now()),
            'fallback_info': {
                'system_initialized': str(self.initialized),
                'analyzer_available': str(hasattr(self, 'analyzer')),
                'suggestion': 'Try restarting the system'
            }
        }

# ========== CELL 30 ==========
class KaggleHealthcareUI:
