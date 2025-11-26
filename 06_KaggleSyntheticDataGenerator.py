class KaggleSyntheticDataGenerator:
  """
  Generates synthetic healthcare data specifically based on Kaggle dataset patterns
  """

  def __init__(self, vector_manager: KagglePineconeVectorManager, analyzer: KaggleDatasetAnalyzer):
    self.vector_manager = vector_manager
    self.analyzer = analyzer
    self.llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0.7,
        max_tokens=3000,
        anthropic_api_key=os.getenv('ANTHROPIC_API_KEY')
    )

  def generate_synthetic_data(self, query: str, num_records: int = 5) -> Dict[str, Any]:
    """
    Generate synthetic data based on learned Kaggle dataset patterns
    """
    try:
      logger.info(f"Generating synthetic data for: {query}")

      # Search for similar patterns
      similar_patterns = self.vector_manager.search_similar_patterns(query, top_k=15)

      # Build context from patterns
      context = self.build_pattern_context(similar_patterns)

      # Create specialized prompt for Kaggle dataset
      prompt = self.create_kaggle_specialized_prompt(query, num_records, context)

      # # Generate data
      response = self.llm.invoke(prompt)

      # Validate and enhance the generated data
      validated_data = self.validate_kaggle_format(response.content, num_records)

      result = {
          'success': True,
          'synthetic_data': validated_data,
          'query': query,
          'num_records': num_records,
          'pattern_sources': len(similar_patterns),
          'timestamp': datetime.now().isoformat(),
          'dataset_source': 'kaggle_healthcare_patterns'
      }

      return result

      # Generate data with retry mechanism
      # max_attempts = 3
      # for attempt in range(max_attempts):
      #     print(f"🎯 Generation attempt {attempt + 1}/{max_attempts} for {num_records} records...")

      #     try:
      #         response = self.llm.invoke(prompt)

      #         # Validate and enhance the generated data
      #         validated_data = self.validate_kaggle_format(response.content, num_records)

      #         # Ensure exact record count
      #         final_data = self.ensure_exact_record_count(validated_data, num_records)

      #         # Verify final count
      #         final_lines = final_data.strip().split('\n')
      #         data_lines = [line for line in final_lines if line.strip() and not ('Name,Age,Gender' in line)]
      #         actual_count = len(data_lines)

      #         if actual_count == num_records:
      #             print(f"✅ Successfully generated exactly {num_records} records!")

      #             result = {
      #                 'success': True,
      #                 'synthetic_data': final_data,
      #                 'query': query,
      #                 'num_records': actual_count,
      #                 'requested_records': num_records,
      #                 'pattern_sources': len(similar_patterns),
      #                 'timestamp': datetime.now().isoformat(),
      #                 'dataset_source': 'kaggle_healthcare_patterns'
      #             }

      #             return result
      #         else:
      #             print(f"⚠️ Attempt {attempt + 1}: Got {actual_count} records, need {num_records}")

      #     except Exception as e:
      #         print(f"❌ Attempt {attempt + 1} failed: {str(e)}")
      #         continue

      # # If all attempts failed, use fallback
      # print(f"🔄 All LLM attempts failed, using enhanced fallback generator...")
      # fallback_data = self.generate_fallback_data(query, num_records)

      # result = {
      #     'success': True,
      #     'synthetic_data': fallback_data,
      #     'query': query,
      #     'num_records': num_records,
      #     'requested_records': num_records,
      #     'pattern_sources': len(similar_patterns),
      #     'timestamp': datetime.now().isoformat(),
      #     'dataset_source': 'enhanced_fallback_generator',
      #     'generation_method': 'fallback'
      # }

      # return result

    except Exception as e:
      logger.error(f"Error generating synthetic data: {str(e)}")
      return {
          'success': False,
          'error': str(e),
          'synthetic_data': None
      }

  def build_pattern_context(self, similar_patterns: List[Dict]) -> str:
    """
    Build context from similar patterns found in the dataset
    """
    context_parts = []

    for pattern in similar_patterns[:5]:  # Use top 5 patterns
      if 'content' in pattern['metadata']:
        # Extract key information from the pattern
        content = pattern['metadata']['content']
        # Anonymize any remaining identifiers
        anonymized_content = self.anonymize_content(content)
        context_parts.append(anonymized_content[:300])  # Limit length

    return "\n".join(context_parts)

  def anonymize_content(self, content: str) -> str:
    """
    Remove any potential identifiers from context content
    """
    import re
    # Replace potential names and specific identifiers
    anonymized = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', 'PATIENT_NAME', content)
    anonymized = re.sub(r'Dr\. [A-Z][a-z]+', 'Dr. DOCTOR_NAME', anonymized)
    anonymized = re.sub(r'\b\d{6,}\b', 'ID_NUMBER', anonymized)
    return anonymized

  def create_kaggle_specialized_prompt(self, query: str, num_records: int, context: str) -> str:
    """
    Create a specialized prompt that leverages learned Kaggle dataset patterns
    """
    # Get distributions for realistic data generation
    age_stats = self.analyzer.distributions['Age']
    condition_probs = self.analyzer.distributions['Medical Condition']['probabilities']
    medication_probs = self.analyzer.distributions['Medication']['probabilities']

    # Get top conditions and medications
    top_conditions = list(self.analyzer.distributions['Medical Condition']['top_values'].keys())[:10]
    top_medications = list(self.analyzer.distributions['Medication']['top_values'].keys())[:10]

    # Get realistic value ranges
    billing_stats = self.analyzer.distributions['Billing Amount']
    room_stats = self.analyzer.distributions['Room Number']

    prompt = f"""You are an expert healthcare data scientist who has analyzed the Kaggle healthcare dataset extensively.
Generate exactly {num_records} realistic synthetic patient records that match the learned patterns and distributions.

DATASET ANALYSIS INSIGHTS:
- Age Range: {age_stats['min']}-{age_stats['max']} years (Mean: {age_stats['mean']:.1f})
- Most Common Conditions: {', '.join(top_conditions[:5])}
- Most Common Medications: {', '.join(top_medications[:5])}
- Billing Range: ${billing_stats['min']:.0f}-${billing_stats['max']:.0f}
- Room Numbers: {room_stats['min']}-{room_stats['max']}

LEARNED PATTERN CONTEXT:
{context}

QUERY FOCUS: {query}

STRICT CSV FORMAT REQUIREMENTS:
1. First line MUST be exactly: Name,Age,Gender,Blood Type,Medical Condition,Date of Admission,Doctor,Hospital,Insurance Provider,Billing Amount,Room Number,Admission Type,Discharge Date,Medication,Test Results
2. Generate exactly {num_records} data rows
3. Each row must have exactly 15 comma-separated values
4. Follow realistic patterns from the dataset analysis

COLUMN SPECIFICATIONS WITH LEARNED PATTERNS:
- Name: Realistic patient names (First Last format)
- Age: Integer {age_stats['min']}-{age_stats['max']} (follow normal distribution around {age_stats['mean']:.0f})
- Gender: "Male" or "Female" (balanced distribution)
- Blood Type: A+, A-, B+, B-, O+, O-, AB+, AB- (O+ most common ~40%, AB- rarest ~1%)
- Medical Condition: Use from [{', '.join(top_conditions[:8])}] based on query context
- Date of Admission: 2023-2024 dates in YYYY-MM-DD format
- Doctor: "Dr. LastName" format (vary specialties based on conditions)
- Hospital: Realistic hospital names with "Hospital", "Medical Center", "Clinic" suffixes
- Insurance Provider: Aetna, Blue Cross, Cigna, UnitedHealth, Medicare, Medicaid
- Billing Amount: ${billing_stats['min']:.0f}-${billing_stats['max']:.0f} (match condition severity)
- Room Number: {room_stats['min']}-{room_stats['max']} (floor-based: 100s, 200s, 300s, etc.)
- Admission Type: Emergency (30%), Elective (50%), Urgent (20%)
- Discharge Date: 1-14 days after admission (vary by condition severity)
- Medication: Match conditions realistically from [{', '.join(top_medications[:10])}]
- Test Results: ONLY use "Normal", "Abnormal", or "Inconclusive" - NO OTHER VALUES ALLOWED

RELATIONSHIP PATTERNS TO FOLLOW:
{self.get_relationship_guidance()}

STRICT INSTRUCTIONS: The Test Results column must contain ONLY these three exact words:
- "Normal" (60% of cases)
- "Abnormal" (35% of cases)
- "Inconclusive" (5% of cases)

Generate the CSV data starting with the header:
Name,Age,Gender,Blood Type,Medical Condition,Date of Admission,Doctor,Hospital,Insurance Provider,Billing Amount,Room Number,Admission Type,Discharge Date,Medication,Test Results"""

    return prompt

  def get_relationship_guidance(self) -> str:
    """
    Get guidance on relationships between columns based on analysis
    """
    guidance_parts = []

    # Age-condition relationships
    if hasattr(self.analyzer, 'relationships') and 'Age_MedicalCondition' in self.analyzer.relationships:
      guidance_parts.append("AGE-CONDITION PATTERNS:")
      for condition, age_info in list(self.analyzer.relationships['Age_MedicalCondition'].items())[:5]:
        guidance_parts.append(f"- {condition}: typically {age_info['mean_age']:.0f} years old")

    # Condition-medication relationships
    if 'Condition_Medication' in self.analyzer.relationships:
      guidance_parts.append("\nCONDITION-MEDICATION PATTERNS:")
      for condition, medications in list(self.analyzer.relationships['Condition_Medication'].items())[:5]:
        top_med = list(medications.keys())[0] if medications else "Various"
        guidance_parts.append(f"- {condition}: commonly treated with {top_med}")

    # Billing patterns
    if 'Billing_Condition' in self.analyzer.relationships:
      guidance_parts.append("\nBILLING PATTERNS:")
      for condition, billing_info in list(self.analyzer.relationships['Billing_Condition'].items())[:5]:
        guidance_parts.append(f"- {condition}: average ${billing_info['mean']:.0f}")

    return "\n".join(guidance_parts)

  def validate_kaggle_format(self, raw_data: str, num_records: int) -> str:
    """
    Validate and fix the generated data with proper indentation
    """
    try:
        # Clean the response
        lines = raw_data.strip().split('\n')
        csv_lines = []

        # Find the header line
        header_found = False
        expected_header = "Name,Age,Gender,Blood Type,Medical Condition,Date of Admission,Doctor,Hospital,Insurance Provider,Billing Amount,Room Number,Admission Type,Discharge Date,Medication,Test Results"

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip markdown or explanatory text
            if line.startswith('#') or line.startswith('```') or line.startswith('*'):
                continue

            # Look for header
            if not header_found and 'Name' in line and 'Age' in line and 'Gender' in line:
                csv_lines.append(expected_header)
                header_found = True
                continue

            # After header, collect data rows
            if header_found and ',' in line:
              # Basic validation: should have 15 comma-separated values
              parts = line.split(',')
              if len(parts) == 15:
                csv_lines.append(line)
                if len(csv_lines) > num_records:  # Header + num_records
                  break

        # If we have valid CSV, return it
        if len(csv_lines) >= 2:  # Header + at least 1 data row
          return '\n'.join(csv_lines)
        else:
          # Generate fallback data using patterns
          return self.generate_fallback_data(num_records)

            # # After header, collect data rows
            # if header_found and ',' in line:
            #     # Basic validation: should have 15 comma-separated values
            #     parts = line.split(',')
            #     if len(parts) == 15:
            #         csv_lines.append(line)
            #     elif len(parts) > 15:
            #         # Try to fix by joining extra parts
            #         print(f"⚠️ Line has {len(parts)} fields, attempting to fix...")
            #         # This is a simple fix - you might need more sophisticated logic
            #         fixed_parts = parts[:14] + [','.join(parts[14:])]
            #         if len(fixed_parts) == 15:
            #             csv_lines.append(','.join(fixed_parts))
            #         else:
            #             print(f"⚠️ Skipping line with {len(parts)} fields (expected 15)")
            #     else:
            #         print(f"⚠️ Skipping line with {len(parts)} fields (expected 15)")

            #     # Check if we have enough rows
            #     if len(csv_lines) > num_records:  # Header + num_records
            #         break

        # # Check if we have enough valid rows
        # data_rows = len(csv_lines) - 1 if csv_lines else 0

        # if data_rows >= num_records:
        #     final_csv = [csv_lines[0]] + csv_lines[1:num_records+1]
        #     print(f"✅ Validated CSV: exactly {num_records} data rows")
        #     return '\n'.join(final_csv)
        # elif data_rows > 0:
        #     missing_count = num_records - data_rows
        #     print(f"🔧 Found {data_rows} valid rows, generating {missing_count} more...")

        #     additional_data = self.generate_fallback_data(missing_count)
        #     additional_lines = additional_data.strip().split('\n')[1:]

        #     final_csv = csv_lines + additional_lines[:missing_count]
        #     print(f"✅ Combined CSV: exactly {num_records} data rows")
        #     return '\n'.join(final_csv)
        # else:
        #     print(f"🔧 No valid rows found, using full fallback...")
        #     return self.generate_fallback_data(num_records)

    except Exception as e:
        logger.warning(f"Validation failed, generating fallback: {e}")
        return self.generate_fallback_data(num_records)

  def generate_fallback_data(self, num_records: int) -> str:
    """
    Generate fallback data using learned patterns when LLM output fails
    """
    import random
    from faker import Faker
    fake = Faker()

    # Use learned distributions
    conditions = list(self.analyzer.distributions['Medical Condition']['top_values'].keys())
    medications = list(self.analyzer.distributions['Medication']['top_values'].keys())

    # Get realistic ranges
    age_min = self.analyzer.distributions['Age']['min']
    age_max = self.analyzer.distributions['Age']['max']
    age_mean = self.analyzer.distributions['Age']['mean']

    billing_min = self.analyzer.distributions['Billing Amount']['min']
    billing_max = self.analyzer.distributions['Billing Amount']['max']

    room_min = self.analyzer.distributions['Room Number']['min']
    room_max = self.analyzer.distributions['Room Number']['max']

    rows = []
    header = "Name,Age,Gender,Blood Type,Medical Condition,Date of Admission,Doctor,Hospital,Insurance Provider,Billing Amount,Room Number,Admission Type,Discharge Date,Medication,Test Results"
    rows.append(header)

    for i in range(num_records):
      # if (i + 1) % 10 == 0:
      #   print(f"📝 Generated {i + 1}/{num_records} records...")
      # Generate realistic data using patterns
      name = fake.name()
      age = max(age_min, min(age_max, int(np.random.normal(age_mean, 15))))
      gender = random.choice(['Male', 'Female'])
      blood_type = random.choices(
            ['O+', 'O-', 'A+', 'A-', 'B+', 'B-', 'AB+', 'AB-'],
            weights=[40, 7, 34, 6, 8, 2, 3, 1]
        )[0]

      condition = random.choice(conditions[:10])

      # Use relationship patterns for medication
      if condition in self.analyzer.relationships.get('Condition_Medication', {}):
        condition_meds = list(self.analyzer.relationships['Condition_Medication'][condition].keys())
        medication = random.choice(condition_meds[:3]) if condition_meds else random.choice(medications[:10])
      else:
        medication = random.choice(medications[:10])

      # Generate dates
      admission_date = fake.date_between(start_date='-1y', end_date='today').strftime('%Y-%m-%d')
      discharge_date = fake.date_between(start_date=admission_date, end_date='+30d').strftime('%Y-%m-%d')

      doctor = f"Dr. {fake.last_name()}"
      hospital = f"{fake.last_name()} {random.choice(['Hospital', 'Medical Center', 'Clinic'])}"
      insurance = random.choice(['Aetna', 'Blue Cross', 'Cigna', 'UnitedHealth', 'Medicare', 'Medicaid'])

      # Realistic billing based on condition
      if condition in self.analyzer.relationships.get('Billing_Condition', {}):
        billing_info = self.analyzer.relationships['Billing_Condition'][condition]
        billing = round(np.random.normal(billing_info['mean'], billing_info['mean'] * 0.3), 2)
        billing = max(billing_min, min(billing_max, billing))
      else:
        billing = round(random.uniform(billing_min, billing_max), 2)

      room = random.randint(room_min, room_max)
      admission_type = random.choices(['Emergency', 'Elective', 'Urgent'], weights=[30, 50, 20])[0]
      test_result = random.choices(['Normal', 'Abnormal', 'Inconclusive'], weights=[60, 35, 5])[0]

      row = f"{name},{age},{gender},{blood_type},{condition},{admission_date},{doctor},{hospital},{insurance},{billing},{room},{admission_type},{discharge_date},{medication},{test_result}"
      rows.append(row)

    print(f"✅ Enhanced fallback generated exactly {num_records} records")
    return '\n'.join(rows)

# ========== CELL 28 ==========
# Updated Enhanced Data Generator with Medical Accuracy for OPTIMISATION
class MedicallyAccurateDataGenerator(KaggleSyntheticDataGenerator):
