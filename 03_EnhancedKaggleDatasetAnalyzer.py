class EnhancedKaggleDatasetAnalyzer:
  """
  Enhanced analyzer that discovers actual relationships in the Kaggle dataset
  """

  def __init__(self, dataset_path: str):
    self.dataset_path = dataset_path
    self.df = None
    self.condition_medication_map = {}
    self.condition_bloodtype_patterns = {}
    self.condition_testresult_patterns = {}
    self.medication_effectiveness_map = {}
    self.realistic_combinations = {}
    # # ADD THIS LINE:
    self.distributions = {}  # Add distributions attribute
    self.patterns = {}       # Add patterns attribute
    self.relationships = {}  # Add relationships attribute

  def load_and_deep_analyze(self):
    """Load dataset and perform deep relationship analysis"""
    logger.info("Loading dataset for deep relationship analysis...")

    self.df = pd.read_csv(self.dataset_path)
    logger.info(f"Loaded {len(self.df)} records for analysis")

    # Create basic distributions
    self._create_basic_distributions()

    # Extract actual values for realistic generation
    self.extract_actual_dataset_values()

    # Perform deep analysis
    self._analyze_condition_medication_relationships()
    self._analyze_condition_bloodtype_patterns()
    self._analyze_condition_testresult_patterns()
    self._analyze_realistic_combinations()
    self._create_medical_accuracy_rules()

    logger.info("✅ Deep relationship analysis complete")
    return self.get_analysis_summary()

  def _analyze_condition_medication_relationships(self):
    """Discover actual condition-medication pairs in the dataset"""
    logger.info("Analyzing condition-medication relationships...")

    # Group by medical condition and analyze medications
    condition_groups = self.df.groupby('Medical Condition')

    for condition, group in condition_groups:
      medication_counts = group['Medication'].value_counts()
      total_patients = len(group)

      # Calculate medication frequencies for this condition
      medication_stats = {}
      for medication, count in medication_counts.items():
          percentage = (count / total_patients) * 100
          medication_stats[medication] = {
              'count': count,
              'percentage': percentage,
              'patients': total_patients
          }

      self.condition_medication_map[condition] = medication_stats

      # Print the actual relationships found
      print(f"\n🔍 {condition}:")
      for medication, stats in list(medication_stats.items())[:5]:  # Top 5
          print(f"  • {medication}: {stats['count']} patients ({stats['percentage']:.1f}%)")

  def _analyze_condition_bloodtype_patterns(self):
    """Analyze blood type patterns for each condition"""
    logger.info("Analyzing condition-blood type patterns...")

    condition_groups = self.df.groupby('Medical Condition')

    for condition, group in condition_groups:
      bloodtype_counts = group['Blood Type'].value_counts()
      total_patients = len(group)

      bloodtype_stats = {}
      for bloodtype, count in bloodtype_counts.items():
          percentage = (count / total_patients) * 100
          bloodtype_stats[bloodtype] = {
              'count': count,
              'percentage': percentage
          }

      self.condition_bloodtype_patterns[condition] = bloodtype_stats

  def _analyze_condition_testresult_patterns(self):
    """Analyze test result patterns for each condition"""
    logger.info("Analyzing condition-test result patterns...")

    condition_groups = self.df.groupby('Medical Condition')

    for condition, group in condition_groups:
      testresult_counts = group['Test Results'].value_counts()
      total_patients = len(group)

      testresult_stats = {}
      for result, count in testresult_counts.items():
          percentage = (count / total_patients) * 100
          testresult_stats[result] = {
              'count': count,
              'percentage': percentage
          }

      self.condition_testresult_patterns[condition] = testresult_stats

      # Print test result patterns
      print(f"\n🧪 Test Results for {condition}:")
      for result, stats in testresult_stats.items():
          print(f"  • {result}: {stats['count']} patients ({stats['percentage']:.1f}%)")

  def _analyze_realistic_combinations(self):
    """Find realistic combinations of condition + medication + test result"""
    logger.info("Analyzing realistic medical combinations...")

    # Group by condition, medication, and test result combinations
    combinations = self.df.groupby(['Medical Condition', 'Medication', 'Test Results']).size().reset_index(name='count')

    # Filter combinations that appear multiple times (more realistic)
    realistic_combinations = combinations[combinations['count'] >= 2]  # At least 2 occurrences

    for _, row in realistic_combinations.iterrows():
      condition = row['Medical Condition']
      medication = row['Medication']
      test_result = row['Test Results']
      count = row['count']

      if condition not in self.realistic_combinations:
          self.realistic_combinations[condition] = []

      self.realistic_combinations[condition].append({
          'medication': medication,
          'test_result': test_result,
          'frequency': count,
          'combination_key': f"{medication}+{test_result}"
      })

    # Sort by frequency for each condition
    for condition in self.realistic_combinations:
        self.realistic_combinations[condition].sort(key=lambda x: x['frequency'], reverse=True)

  def _create_medical_accuracy_rules(self):
    """Create rules for medically accurate data generation"""
    logger.info("Creating medical accuracy rules...")

    # Create medication effectiveness mapping based on test results
    for condition, combinations in self.realistic_combinations.items():
      if condition not in self.medication_effectiveness_map:
        self.medication_effectiveness_map[condition] = {}

      for combo in combinations:
        medication = combo['medication']
        test_result = combo['test_result']
        frequency = combo['frequency']

        if medication not in self.medication_effectiveness_map[condition]:
            self.medication_effectiveness_map[condition][medication] = {
                'Normal': 0, 'Abnormal': 0, 'Inconclusive': 0
            }

        self.medication_effectiveness_map[condition][medication][test_result] += frequency

  def get_realistic_medication_for_condition(self, condition: str, prefer_common: bool = True):
    """Get realistic medication based on actual dataset patterns"""
    if condition not in self.condition_medication_map:
      # Fallback to most common medications in dataset
      all_medications = self.df['Medication'].value_counts()
      return list(all_medications.index)[:5]

    medications = self.condition_medication_map[condition]

    if prefer_common:
      # Return medications that appear in at least 5% of cases for this condition
      realistic_meds = [med for med, stats in medications.items()
                        if stats['percentage'] >= 3.0]  # At least 3% of cases

      if realistic_meds:
          return realistic_meds

    # Return top medications for this condition
    return list(medications.keys())[:5]

  def get_realistic_test_result_for_condition_medication(self, condition: str, medication: str):
    """Get realistic test result based on condition-medication combination"""
    if condition in self.realistic_combinations:
      # Find combinations with this condition and medication
      relevant_combos = [combo for combo in self.realistic_combinations[condition]
                        if combo['medication'] == medication]

      if relevant_combos:
          # Weight by frequency
          total_freq = sum(combo['frequency'] for combo in relevant_combos)

          # Create weighted probability
          test_result_probs = {}
          for combo in relevant_combos:
              result = combo['test_result']
              prob = combo['frequency'] / total_freq
              test_result_probs[result] = test_result_probs.get(result, 0) + prob

          return test_result_probs

    # Fallback to general condition patterns
    if condition in self.condition_testresult_patterns:
      condition_results = self.condition_testresult_patterns[condition]
      return {result: stats['percentage']/100 for result, stats in condition_results.items()}

    # Default probabilities
    return {'Normal': 0.6, 'Abnormal': 0.35, 'Inconclusive': 0.05}

  def get_analysis_summary(self):
    """Get comprehensive analysis summary"""
    return {
        'condition_medication_map': self.condition_medication_map,
        'condition_bloodtype_patterns': self.condition_bloodtype_patterns,
        'condition_testresult_patterns': self.condition_testresult_patterns,
        'realistic_combinations': self.realistic_combinations,
        'medication_effectiveness_map': self.medication_effectiveness_map,
        'total_conditions': len(self.condition_medication_map),
        'total_realistic_combinations': sum(len(combos) for combos in self.realistic_combinations.values())
    }

  def print_medical_accuracy_report(self):
    """Print detailed medical accuracy report"""
    print("\n" + "="*80)
    print("🏥 MEDICAL ACCURACY ANALYSIS REPORT")
    print("="*80)

    print(f"\n📊 DATASET OVERVIEW:")
    print(f"• Total Records: {len(self.df):,}")
    print(f"• Unique Conditions: {self.df['Medical Condition'].nunique()}")
    print(f"• Unique Medications: {self.df['Medication'].nunique()}")
    print(f"• Unique Blood Types: {self.df['Blood Type'].nunique()}")

    print(f"\n💊 CONDITION-MEDICATION ACCURACY:")
    for condition, medications in self.condition_medication_map.items():
      print(f"\n🔸 {condition}:")

      # Show top 3 medications for each condition
      top_meds = sorted(medications.items(), key=lambda x: x[1]['percentage'], reverse=True)[:3]
      for med, stats in top_meds:
          print(f"   ✓ {med}: {stats['count']} patients ({stats['percentage']:.1f}%)")

    print(f"\n🧪 HIGH-CONFIDENCE MEDICAL COMBINATIONS:")
    for condition, combinations in self.realistic_combinations.items():
      if combinations:  # Only show conditions with realistic combinations
          print(f"\n🔸 {condition}:")
          # Show top 3 combinations
          for combo in combinations[:3]:
              print(f"   ✓ {combo['medication']} → {combo['test_result']} (seen {combo['frequency']} times)")

    print(f"\n⚠️ POTENTIAL ISSUES IDENTIFIED:")
    issues_found = 0

    # Check for unrealistic medication-condition pairs
    medical_knowledge = {
        'Asthma': ['Albuterol', 'Inhaler', 'Bronchodilator', 'Corticosteroid'],
        'Diabetes': ['Insulin', 'Metformin', 'Glipizide', 'Glucophage'],
        'Hypertension': ['Lisinopril', 'Amlodipine', 'Hydrochlorothiazide', 'Losartan'],
        'Arthritis': ['Ibuprofen', 'Naproxen', 'Celecoxib', 'Prednisone'],
        'Cancer': ['Chemotherapy', 'Radiation', 'Immunotherapy', 'Targeted therapy']
    }

    for condition, actual_medications in self.condition_medication_map.items():
      if condition in medical_knowledge:
        expected_meds = medical_knowledge[condition]
        actual_med_names = list(actual_medications.keys())

        # Check if any expected medications are present
        matches = [med for med in expected_meds if any(expected.lower() in med.lower() for expected in expected_meds)]

        if not matches:
          issues_found += 1
          print(f"   ⚠️ {condition}: No typical medications found")
          print(f"      Expected: {', '.join(expected_meds)}")
          print(f"      Found: {', '.join(list(actual_medications.keys())[:3])}")

    if issues_found == 0:
      print("   ✅ No major medical accuracy issues detected!")

    return issues_found

  # # for fixing the UI issue on the number of records
  def _create_basic_distributions(self):
    """Create basic distributions like the original analyzer"""
    # Age distribution
    self.distributions['Age'] = {
        'min': int(self.df['Age'].min()),
        'max': int(self.df['Age'].max()),
        'mean': float(self.df['Age'].mean()),
        'std': float(self.df['Age'].std()),
        'quartiles': [float(q) for q in self.df['Age'].quantile([0.25, 0.5, 0.75])]
    }

    # Billing Amount distribution
    self.distributions['Billing Amount'] = {
        'min': float(self.df['Billing Amount'].min()),
        'max': float(self.df['Billing Amount'].max()),
        'mean': float(self.df['Billing Amount'].mean()),
        'std': float(self.df['Billing Amount'].std()),
        'quartiles': [float(q) for q in self.df['Billing Amount'].quantile([0.25, 0.5, 0.75])]
    }

    # Room Number distribution
    self.distributions['Room Number'] = {
        'min': int(self.df['Room Number'].min()),
        'max': int(self.df['Room Number'].max())
    }

    # Medical Condition distribution
    self.distributions['Medical Condition'] = {
        'value_counts': self.df['Medical Condition'].value_counts().to_dict(),
        'top_values': self.df['Medical Condition'].value_counts().head(10).to_dict()
    }

    # Medication distribution
    self.distributions['Medication'] = {
        'value_counts': self.df['Medication'].value_counts().to_dict(),
        'top_values': self.df['Medication'].value_counts().head(10).to_dict()
    }
  # # for fixing the UI issue
  def extract_actual_dataset_values(self):
    """Extract all actual values from the dataset for realistic generation"""
    print("🔍 Extracting actual dataset values...")

    self.actual_values = {
        'names': list(self.df['Name'].unique()),
        'doctors': list(self.df['Doctor'].unique()),
        'hospitals': list(self.df['Hospital'].unique()),
        'insurance_providers': list(self.df['Insurance Provider'].unique()),
        'medications': list(self.df['Medication'].unique()),
        'medical_conditions': list(self.df['Medical Condition'].unique()),
        'blood_types': list(self.df['Blood Type'].unique()),
        'admission_types': list(self.df['Admission Type'].unique()),
        'test_results': list(self.df['Test Results'].unique()),
        'genders': list(self.df['Gender'].unique())
    }

    # Extract date ranges from actual data
    self.df['Date of Admission'] = pd.to_datetime(self.df['Date of Admission'])
    self.df['Discharge Date'] = pd.to_datetime(self.df['Discharge Date'])

    self.actual_values['admission_date_range'] = {
        'min': self.df['Date of Admission'].min(),
        'max': self.df['Date of Admission'].max()
    }

    self.actual_values['discharge_date_range'] = {
        'min': self.df['Discharge Date'].min(),
        'max': self.df['Discharge Date'].max()
    }

    # Extract numerical ranges
    self.actual_values['age_range'] = {
        'min': int(self.df['Age'].min()),
        'max': int(self.df['Age'].max()),
        'values': list(self.df['Age'].unique())
    }

    self.actual_values['room_range'] = {
        'min': int(self.df['Room Number'].min()),
        'max': int(self.df['Room Number'].max()),
        'values': list(self.df['Room Number'].unique())
    }

    self.actual_values['billing_range'] = {
        'min': float(self.df['Billing Amount'].min()),
        'max': float(self.df['Billing Amount'].max()),
        'values': list(self.df['Billing Amount'].unique())
    }

    print(f"✅ Extracted values from {len(self.df)} actual records")
    return self.actual_values

# ========== CELL 25 ==========
class PrivacySecurityManager:
