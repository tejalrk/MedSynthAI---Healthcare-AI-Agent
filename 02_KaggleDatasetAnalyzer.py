class KaggleDatasetAnalyzer:
  """
  Analyzes the Kaggle healthcare dataset to learn patterns and distributions
  """

  def __init__(self, dataset_path: str):
    self.dataset_path = dataset_path
    self.df = None
    self.patterns = {}
    self.distributions = {}
    self.relationships = {}

  def load_and_analyze(self):
    """Load the dataset and perform comprehensive analysis"""
    logger.info("Loading and analyzing Kaggle healthcare dataset...")

    # Load dataset
    self.df = pd.read_csv(self.dataset_path)
    logger.info(f"Loaded dataset: {self.df.shape[0]} rows, {self.df.shape[1]} columns")

    # Analyze each column
    self._analyze_distributions()
    self._analyze_relationships()
    self._analyze_patterns()

    logger.info("✅ Dataset analysis complete")
    return self.patterns, self.distributions, self.relationships

  def _analyze_distributions(self):
    """Analyze distributions for each column"""
    logger.info("Analyzing column distributions...")

    # Age distribution
    self.distributions['Age'] = {
        'min': int(self.df['Age'].min()),
        'max': int(self.df['Age'].max()),
        'mean': float(self.df['Age'].mean()),
        'std': float(self.df['Age'].std()),
        'quartiles': [float(q) for q in self.df['Age'].quantile([0.25, 0.5, 0.75])],
        'common_ranges': self._get_age_ranges()
    }

    # Categorical distributions
    categorical_columns = ['Gender', 'Blood Type', 'Medical Condition', 'Doctor',
                          'Hospital', 'Insurance Provider', 'Admission Type', 'Medication', 'Test Results']

    for col in categorical_columns:
      self.distributions[col] = {
            'unique_count': int(self.df[col].nunique()),
            'value_counts': self.df[col].value_counts().to_dict(),
            'top_values': self.df[col].value_counts().head(10).to_dict(),
            'probabilities': (self.df[col].value_counts(normalize=True)).to_dict()
        }

    # Billing Amount distribution
    self.distributions['Billing Amount'] = {
        'min': float(self.df['Billing Amount'].min()),
        'max': float(self.df['Billing Amount'].max()),
        'mean': float(self.df['Billing Amount'].mean()),
        'std': float(self.df['Billing Amount'].std()),
        'quartiles': [float(q) for q in self.df['Billing Amount'].quantile([0.25, 0.5, 0.75])],
        'ranges_by_condition': self._get_billing_by_condition()
    }

    # Room Number distribution
    self.distributions['Room Number'] = {
        'min': int(self.df['Room Number'].min()),
        'max': int(self.df['Room Number'].max()),
        'common_ranges': self._get_room_ranges()
    }

  def _analyze_relationships(self):
    """Analyze relationships between columns"""
    logger.info("Analyzing column relationships...")

    # Age vs Medical Condition
    self.relationships['Age_MedicalCondition'] = {}
    for condition in self.df['Medical Condition'].unique():
      condition_ages = self.df[self.df['Medical Condition'] == condition]['Age']
      self.relationships['Age_MedicalCondition'][condition] = {
            'mean_age': float(condition_ages.mean()),
            'age_range': [int(condition_ages.min()), int(condition_ages.max())],
            'common_age_group': self._get_age_group(condition_ages.mean())
        }

    # Medical Condition vs Medication
    self.relationships['Condition_Medication'] = {}
    for condition in self.df['Medical Condition'].unique():
      medications = self.df[self.df['Medical Condition'] == condition]['Medication'].value_counts()
      self.relationships['Condition_Medication'][condition] = medications.head(5).to_dict()

    # Medical Condition vs Test Results
    self.relationships['Condition_TestResults'] = {}
    for condition in self.df['Medical Condition'].unique():
      test_results = self.df[self.df['Medical Condition'] == condition]['Test Results'].value_counts(normalize=True)
      self.relationships['Condition_TestResults'][condition] = test_results.to_dict()

    # Admission Type vs Medical Condition
    self.relationships['AdmissionType_Condition'] = {}
    for admission_type in self.df['Admission Type'].unique():
      conditions = self.df[self.df['Admission Type'] == admission_type]['Medical Condition'].value_counts()
      self.relationships['AdmissionType_Condition'][admission_type] = conditions.head(5).to_dict()

    # Billing Amount vs Medical Condition
    self.relationships['Billing_Condition'] = {}
    for condition in self.df['Medical Condition'].unique():
      condition_billing = self.df[self.df['Medical Condition'] == condition]['Billing Amount']
      self.relationships['Billing_Condition'][condition] = {
            'mean': float(condition_billing.mean()),
            'median': float(condition_billing.median()),
            'range': [float(condition_billing.min()), float(condition_billing.max())]
        }

  def _analyze_patterns(self):
    """Analyze common patterns in the data"""
    logger.info("Analyzing data patterns...")

    # Name patterns
    self.patterns['Name'] = {
        'first_names': list(set([name.split()[0] for name in self.df['Name'] if len(name.split()) > 0])),
        'last_names': list(set([name.split()[-1] for name in self.df['Name'] if len(name.split()) > 1])),
        'name_formats': ['First Last', 'First Middle Last']
    }

    # Doctor patterns
    self.patterns['Doctor'] = {
        'prefixes': ['Dr.'],
        'specialties_inferred': self._infer_doctor_specialties(),
        'name_patterns': list(set([doc.replace('Dr. ', '') for doc in self.df['Doctor'].unique()]))
    }

    # Hospital patterns
    self.patterns['Hospital'] = {
        'suffixes': ['Hospital', 'Medical Center', 'Clinic', 'General Hospital'],
        'common_names': list(self.df['Hospital'].unique())
    }

    # Date patterns
    self.patterns['Dates'] = {
        'admission_date_range': [self.df['Date of Admission'].min(), self.df['Date of Admission'].max()],
        'typical_stay_length': self._calculate_stay_patterns(),
        'seasonal_patterns': self._analyze_seasonal_admissions()
    }

  def _get_age_ranges(self):
    """Get common age ranges"""
    ranges = {
        '18-30': len(self.df[(self.df['Age'] >= 18) & (self.df['Age'] <= 30)]),
        '31-45': len(self.df[(self.df['Age'] >= 31) & (self.df['Age'] <= 45)]),
        '46-60': len(self.df[(self.df['Age'] >= 46) & (self.df['Age'] <= 60)]),
        '61-75': len(self.df[(self.df['Age'] >= 61) & (self.df['Age'] <= 75)]),
        '76+': len(self.df[self.df['Age'] >= 76])
    }
    return ranges

  def _get_billing_by_condition(self):
    """Get billing patterns by medical condition"""
    billing_by_condition = {}
    for condition in self.df['Medical Condition'].unique():
        condition_billing = self.df[self.df['Medical Condition'] == condition]['Billing Amount']
        billing_by_condition[condition] = {
            'mean': float(condition_billing.mean()),
            'range': [float(condition_billing.min()), float(condition_billing.max())]
        }
    return billing_by_condition

  def _get_room_ranges(self):
    """Get room number patterns"""
    rooms = self.df['Room Number'].unique()
    return {
        'floor_1': len([r for r in rooms if 100 <= r <= 199]),
        'floor_2': len([r for r in rooms if 200 <= r <= 299]),
        'floor_3': len([r for r in rooms if 300 <= r <= 399]),
        'floor_4': len([r for r in rooms if 400 <= r <= 499]),
        'floor_5': len([r for r in rooms if 500 <= r <= 599])
    }

  def _get_age_group(self, age):
    """Categorize age into groups"""
    if age < 30:
      return 'young_adult'
    elif age < 50:
      return 'middle_aged'
    elif age < 70:
      return 'senior'
    else:
      return 'elderly'

  def _infer_doctor_specialties(self):
    """Infer doctor specialties based on common conditions they treat"""
    doctor_conditions = {}
    for _, row in self.df.iterrows():
      doctor = row['Doctor']
      condition = row['Medical Condition']
      if doctor not in doctor_conditions:
          doctor_conditions[doctor] = {}
      doctor_conditions[doctor][condition] = doctor_conditions[doctor].get(condition, 0) + 1

    specialties = {}
    for doctor, conditions in doctor_conditions.items():
      top_condition = max(conditions, key=conditions.get)
      specialties[doctor] = self._map_condition_to_specialty(top_condition)

    return specialties

  def _map_condition_to_specialty(self, condition):
    """Map medical conditions to likely specialties"""
    specialty_mapping = {
        'Diabetes': 'Endocrinology',
        'Hypertension': 'Cardiology',
        'Asthma': 'Pulmonology',
        'Arthritis': 'Rheumatology',
        'Cancer': 'Oncology',
        'Obesity': 'Endocrinology',
        'Migraine': 'Neurology'
    }
    return specialty_mapping.get(condition, 'Internal Medicine')

  def _calculate_stay_patterns(self):
    """Calculate hospital stay length patterns"""
    self.df['Date of Admission'] = pd.to_datetime(self.df['Date of Admission'])
    self.df['Discharge Date'] = pd.to_datetime(self.df['Discharge Date'])
    self.df['Stay Length'] = (self.df['Discharge Date'] - self.df['Date of Admission']).dt.days

    return {
        'mean_stay': float(self.df['Stay Length'].mean()),
        'median_stay': float(self.df['Stay Length'].median()),
        'stay_by_condition': self.df.groupby('Medical Condition')['Stay Length'].mean().to_dict()
    }

  def _analyze_seasonal_admissions(self):
    """Analyze seasonal admission patterns"""
    self.df['Month'] = pd.to_datetime(self.df['Date of Admission']).dt.month
    monthly_admissions = self.df['Month'].value_counts().sort_index()
    return monthly_admissions.to_dict()

# ========== CELL 24 ==========
# FOR OPTIMISATION
class EnhancedKaggleDatasetAnalyzer:
