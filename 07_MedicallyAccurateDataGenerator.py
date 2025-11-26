class MedicallyAccurateDataGenerator(KaggleSyntheticDataGenerator):
  """
  Enhanced generator that uses actual dataset relationships for medical accuracy
  """

  def __init__(self, vector_manager: KagglePineconeVectorManager, analyzer: EnhancedKaggleDatasetAnalyzer):
    super().__init__(vector_manager, analyzer)
    self.enhanced_analyzer = analyzer

  def create_kaggle_specialized_prompt(self, query: str, num_records: int, context: str) -> str:
    """
    Create medically accurate prompt using discovered relationships
    """
    # Get query-specific medical accuracy information
    medical_context = self._extract_medical_context_from_query(query)

    prompt = f"""You are a medical expert generating realistic synthetic healthcare data based on ACTUAL patterns from the Kaggle healthcare dataset.

CRITICAL MEDICAL ACCURACY REQUIREMENTS:
Use ONLY the medication-condition relationships found in the actual dataset below.

{self._get_medical_accuracy_guidance(query)}

QUERY: {query}
Generate exactly {num_records} records.

STRICT CSV FORMAT:
Name,Age,Gender,Blood Type,Medical Condition,Date of Admission,Doctor,Hospital,Insurance Provider,Billing Amount,Room Number,Admission Type,Discharge Date,Medication,Test Results

MEDICAL ACCURACY RULES:
{self._create_medical_accuracy_rules(query)}

Generate realistic data that follows these ACTUAL dataset patterns:"""

    return prompt

  def _extract_medical_context_from_query(self, query: str) -> dict:
    """Extract medical context from the query"""
    query_lower = query.lower()

    # Identify mentioned conditions
    mentioned_conditions = []
    for condition in self.enhanced_analyzer.condition_medication_map.keys():
      if condition.lower() in query_lower:
        mentioned_conditions.append(condition)

    # Identify mentioned medications
    mentioned_medications = []
    for medications in self.enhanced_analyzer.condition_medication_map.values():
      for med in medications.keys():
        if med.lower() in query_lower:
          mentioned_medications.append(med)

    return {
        'conditions': mentioned_conditions,
        'medications': mentioned_medications,
        'query_type': self._determine_query_type(query_lower)
    }

  def _get_medical_accuracy_guidance(self, query: str) -> str:
    """Get specific medical accuracy guidance based on query"""
    guidance_parts = []
    query_lower = query.lower()

    # Check for specific conditions mentioned in query
    for condition, medications in self.enhanced_analyzer.condition_medication_map.items():
      if condition.lower() in query_lower:
        top_meds = sorted(medications.items(), key=lambda x: x[1]['percentage'], reverse=True)[:3]

        guidance_parts.append(f"\nFor {condition} patients, use ONLY these medications found in dataset:")
        for med, stats in top_meds:
          guidance_parts.append(f"• {med} (used by {stats['percentage']:.1f}% of {condition} patients)")

        # Add test result guidance
        if condition in self.enhanced_analyzer.condition_testresult_patterns:
          test_patterns = self.enhanced_analyzer.condition_testresult_patterns[condition]
          guidance_parts.append(f"\nTest Results for {condition}:")
          for result, stats in test_patterns.items():
            guidance_parts.append(f"• {result}: {stats['percentage']:.1f}% probability")

    return '\n'.join(guidance_parts) if guidance_parts else "Use general dataset medication patterns."

  def _create_medical_accuracy_rules(self, query: str) -> str:
    """Create specific medical accuracy rules"""
    rules = []

    # Add condition-specific rules
    for condition in self.enhanced_analyzer.condition_medication_map.keys():
      if condition.lower() in query.lower():
        realistic_meds = self.enhanced_analyzer.get_realistic_medication_for_condition(condition)
        rules.append(f"• {condition} patients must use: {', '.join(realistic_meds[:3])}")

        # Add test result rules
        test_patterns = self.enhanced_analyzer.condition_testresult_patterns.get(condition, {})
        if test_patterns:
          most_common_result = max(test_patterns.items(), key=lambda x: x[1]['percentage'])
          rules.append(f"• {condition} patients most commonly have: {most_common_result[0]} results ({most_common_result[1]['percentage']:.1f}%)")

    if not rules:
      rules.append("• Follow general dataset medication-condition patterns")
      rules.append("• Ensure medical realism based on actual data relationships")

    return '\n'.join(rules)

  def _determine_query_type(self, query_lower: str) -> str:
    """Determine the type of query for targeted generation"""
    if 'medication' in query_lower or 'prescribed' in query_lower:
      return 'medication_focused'
    elif 'test' in query_lower or 'result' in query_lower:
      return 'test_focused'
    elif 'emergency' in query_lower or 'urgent' in query_lower:
      return 'admission_focused'
    else:
      return 'general'

  # # Below function is added to fix the UI issue with the number of records
  # def ensure_exact_record_count(self, csv_data: str, target_count: int) -> str:
  #   """Ensure the CSV has exactly the target number of records"""
  #   lines = csv_data.strip().split('\n')

  #   # Find header and data rows
  #   header_line = None
  #   data_rows = []

  #   for line in lines:
  #       line = line.strip()
  #       if not line:
  #           continue

  #       # Check if this looks like a header
  #       if 'Name' in line and 'Age' in line and 'Gender' in line:
  #           header_line = line
  #       elif ',' in line and len(line.split(',')) == 15:
  #           data_rows.append(line)

  #   if not header_line:
  #       header_line = "Name,Age,Gender,Blood Type,Medical Condition,Date of Admission,Doctor,Hospital,Insurance Provider,Billing Amount,Room Number,Admission Type,Discharge Date,Medication,Test Results"

  #   print(f"🔍 Found {len(data_rows)} records, target: {target_count}")

  #   # If we have enough records, trim to exact count
  #   if len(data_rows) >= target_count:
  #       final_rows = data_rows[:target_count]
  #       print(f"✂️ Trimmed to exactly {target_count} records")

  #   # If we have too few records, generate more using patterns
  #   elif len(data_rows) < target_count:
  #       missing_count = target_count - len(data_rows)
  #       print(f"📝 Generating {missing_count} additional records using fallback...")

  #       # Use the fallback generator for missing records
  #       additional_csv = self.generate_fallback_data(missing_count)
  #       additional_lines = additional_csv.strip().split('\n')[1:]  # Skip header

  #       final_rows = data_rows + additional_lines[:missing_count]
  #       print(f"✅ Completed to exactly {target_count} records")

  #   # Construct final CSV
  #   final_csv = header_line + '\n' + '\n'.join(final_rows)
  #   return final_csv

  # # For fixing the UI issue for the number of records
  # # Override the parent's generate_fallback_data method
  # def generate_fallback_data(self, num_records: int) -> str:
  #   """
  #   Generate fallback data using ONLY actual dataset values with comma safety
  #   """
  #   import random
  #   from datetime import datetime, timedelta

  #   try:
  #     print(f"🔄 Generating {num_records} records using ACTUAL dataset values...")

  #     # Get actual values from enhanced analyzer
  #     actual_values = self.enhanced_analyzer.actual_values
  #     conditions = actual_values['medical_conditions']

  #     rows = []
  #     header = "Name,Age,Gender,Blood Type,Medical Condition,Date of Admission,Doctor,Hospital,Insurance Provider,Billing Amount,Room Number,Admission Type,Discharge Date,Medication,Test Results"
  #     rows.append(header)

  #     # Track used combinations to avoid duplicates
  #     used_combinations = set()

  #     def clean_csv_value(value):
  #       """Remove or replace problematic characters in CSV values"""
  #       if isinstance(value, str):
  #           # Replace commas with semicolons or remove them
  #           value = value.replace(',', ';').replace('\n', ' ').replace('\r', ' ')
  #           # Remove any quotes that might cause issues
  #           value = value.replace('"', '').replace("'", "")
  #           return value.strip()
  #       return str(value)

  #     for i in range(num_records):
  #       max_attempts = 50
  #       attempt = 0

  #       while attempt < max_attempts:
  #         try:
  #             # Use actual names or generate unique ones (clean them)
  #             if len(actual_values['names']) > i:
  #               name = clean_csv_value(actual_values['names'][i % len(actual_values['names'])])
  #             else:
  #               name = f"Patient_{i+1:04d}"

  #             # Use actual values from dataset
  #             age = random.choice(actual_values['age_range']['values'])
  #             gender = clean_csv_value(random.choice(actual_values['genders']))
  #             blood_type = clean_csv_value(random.choice(actual_values['blood_types']))
  #             condition = clean_csv_value(random.choice(conditions))

  #             # Use condition-specific medication if available
  #             if condition.replace(';', ',') in self.enhanced_analyzer.condition_medication_map:
  #               condition_meds = list(self.enhanced_analyzer.condition_medication_map[condition.replace(';', ',')].keys())
  #               medication = clean_csv_value(random.choice(condition_meds)) if condition_meds else clean_csv_value(random.choice(actual_values['medications']))
  #             else:
  #               medication = clean_csv_value(random.choice(actual_values['medications']))

  #             # Clean all text fields
  #             doctor = clean_csv_value(random.choice(actual_values['doctors']))
  #             hospital = clean_csv_value(random.choice(actual_values['hospitals']))
  #             insurance = clean_csv_value(random.choice(actual_values['insurance_providers']))
  #             admission_type = clean_csv_value(random.choice(actual_values['admission_types']))
  #             test_result = clean_csv_value(random.choice(actual_values['test_results']))

  #             # Use actual room numbers and billing amounts
  #             room = int(random.choice(actual_values['room_range']['values']))
  #             billing = float(random.choice(actual_values['billing_range']['values']))

  #             # Generate dates within actual dataset range
  #             admission_date_range = actual_values['admission_date_range']
  #             days_diff = (admission_date_range['max'] - admission_date_range['min']).days
  #             random_days = random.randint(0, max(1, days_diff))
  #             admission_date = (admission_date_range['min'] + timedelta(days=random_days)).strftime('%Y-%m-%d')

  #             # Generate discharge date (1-30 days after admission)
  #             discharge_days = random.randint(1, 30)
  #             admission_dt = datetime.strptime(admission_date, '%Y-%m-%d')
  #             discharge_date = (admission_dt + timedelta(days=discharge_days)).strftime('%Y-%m-%d')

  #             # Create unique combination key to avoid duplicates
  #             combination_key = f"{name}_{age}_{condition}_{medication}_{admission_date}"

  #             if combination_key not in used_combinations:
  #               used_combinations.add(combination_key)

  #               # Build row with exactly 15 fields
  #               row_parts = [
  #                   name,                    # 1. Name
  #                   str(age),               # 2. Age
  #                   gender,                 # 3. Gender
  #                   blood_type,             # 4. Blood Type
  #                   condition,              # 5. Medical Condition
  #                   admission_date,         # 6. Date of Admission
  #                   doctor,                 # 7. Doctor
  #                   hospital,               # 8. Hospital
  #                   insurance,              # 9. Insurance Provider
  #                   str(billing),           # 10. Billing Amount
  #                   str(room),              # 11. Room Number
  #                   admission_type,         # 12. Admission Type
  #                   discharge_date,         # 13. Discharge Date
  #                   medication,             # 14. Medication
  #                   test_result             # 15. Test Results
  #               ]

  #               # Verify we have exactly 15 fields
  #               if len(row_parts) == 15:
  #                 row = ','.join(row_parts)
  #                 rows.append(row)
  #                 break
  #               else:
  #                 print(f"⚠️ Row {i+1} has {len(row_parts)} fields, expected 15")
  #                 attempt += 1
  #                 continue

  #             else:
  #               attempt += 1
  #               continue

  #         except Exception as e:
  #           print(f"⚠️ Error generating row {i+1}, attempt {attempt+1}: {e}")
  #           attempt += 1
  #           continue

  #       # If we couldn't generate a unique combination, generate a simple one
  #       if attempt >= max_attempts:
  #           simple_parts = [
  #               f"UniquePatient_{i+1:04d}",                                    # Name
  #               str(random.choice(actual_values['age_range']['values'])),      # Age
  #               clean_csv_value(random.choice(actual_values['genders'])),     # Gender
  #               clean_csv_value(random.choice(actual_values['blood_types'])), # Blood Type
  #               clean_csv_value(random.choice(conditions)),                   # Medical Condition
  #               admission_date,                                               # Date of Admission
  #               f"Dr.Smith_{i+1}",                                          # Doctor
  #               f"Hospital_{i+1}",                                          # Hospital
  #               clean_csv_value(random.choice(actual_values['insurance_providers'])), # Insurance
  #               str(random.choice(actual_values['billing_range']['values'])), # Billing Amount
  #               str(random.choice(actual_values['room_range']['values'])),    # Room Number
  #               clean_csv_value(random.choice(actual_values['admission_types'])), # Admission Type
  #               discharge_date,                                              # Discharge Date
  #               clean_csv_value(random.choice(actual_values['medications'])), # Medication
  #               clean_csv_value(random.choice(actual_values['test_results'])) # Test Results
  #           ]

  #           if len(simple_parts) == 15:
  #               rows.append(','.join(simple_parts))

  #     result = '\n'.join(rows)

  #     # Final validation - check each line has 15 fields
  #     validation_lines = result.strip().split('\n')
  #     valid_lines = [validation_lines[0]]  # Keep header

  #     for i, line in enumerate(validation_lines[1:], 1):
  #         parts = line.split(',')
  #         if len(parts) == 15:
  #             valid_lines.append(line)
  #         else:
  #             print(f"⚠️ Skipping invalid line {i+1}: {len(parts)} fields")

  #     print(f"✅ Generated {len(valid_lines)-1} valid records using actual dataset values")
  #     print(f"📊 Used {len(used_combinations)} unique combinations")

  #     return '\n'.join(valid_lines)

  #   except Exception as e:
  #       print(f"❌ Error in dataset-aware generation: {e}")
  #       return self._generate_basic_emergency_fallback(num_records)

  # def _generate_basic_emergency_fallback(self, num_records: int) -> str:
  #   """Ultra-simple emergency fallback with guaranteed 15 fields"""
  #   print(f"🚨 Using basic emergency fallback for {num_records} records")

  #   rows = []
  #   header = "Name,Age,Gender,Blood Type,Medical Condition,Date of Admission,Doctor,Hospital,Insurance Provider,Billing Amount,Room Number,Admission Type,Discharge Date,Medication,Test Results"
  #   rows.append(header)

  #   for i in range(num_records):
  #     # Guarantee exactly 15 fields
  #     row_parts = [
  #         f"EmergencyPatient_{i+1:04d}",  # 1. Name
  #         "45",                           # 2. Age
  #         "Male",                         # 3. Gender
  #         "O+",                          # 4. Blood Type
  #         "Diabetes",                     # 5. Medical Condition
  #         "2024-01-15",                  # 6. Date of Admission
  #         f"Dr.Emergency_{i+1}",         # 7. Doctor
  #         f"EmergencyHospital_{i+1}",    # 8. Hospital
  #         "Medicare",                     # 9. Insurance Provider
  #         "25000",                       # 10. Billing Amount
  #         str(301 + i),                  # 11. Room Number
  #         "Emergency",                    # 12. Admission Type
  #         "2024-01-20",                  # 13. Discharge Date
  #         "Metformin",                   # 14. Medication
  #         "Normal"                       # 15. Test Results
  #     ]

  #     rows.append(','.join(row_parts))

  #   return '\n'.join(rows)

  # # Also fix the enhanced fallback method
  # def generate_enhanced_fallback_data(self, query: str, num_records: int) -> str:
  #     """
  #     Generate exact number of records using enhanced patterns
  #     """
  #     # Just call the regular fallback method which is now enhanced
  #     return self.generate_fallback_data(num_records)

# ========== CELL 29 ==========
class KaggleHealthcareAgent:
