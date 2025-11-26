class KagglePineconeVectorManager:
  """
  Manages Pinecone vector database operations specifically for Kaggle healthcare dataset
  """

  def __init__(self, index_name: str = "kaggle-healthcare-v1"):
    self.index_name = index_name
    self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    self.embeddings = VoyageAIEmbeddings(
        model="voyage-3",
        voyage_api_key=os.getenv('VOYAGE_API_KEY')
    )
    self.dataset_analyzer = None
    self.initialize_index()

  def initialize_index(self):
    """Initialize Pinecone index"""
    try:
      existing_indexes = [index.name for index in self.pc.list_indexes()]

      if self.index_name not in existing_indexes:
        logger.info(f"Creating new Pinecone index: {self.index_name}")
        self.pc.create_index(
            name=self.index_name,
            dimensions=1024,  # voyage-3 dimensions
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

      self.index = self.pc.Index(self.index_name)
      logger.info(f"✅ Connected to Pinecone index: {self.index_name}")

    except Exception as e:
      logger.error(f"Error initializing Pinecone index: {str(e)}")
      raise

  def process_kaggle_dataset(self, df: pd.DataFrame, analyzer: KaggleDatasetAnalyzer):
    """
    Process the Kaggle healthcare dataset and store in Pinecone
    """
    try:
        logger.info("Processing Kaggle healthcare dataset...")
        self.dataset_analyzer = analyzer

        # Sample for faster processing if dataset is large
        if len(df) > 2000:
          logger.info(f"Sampling dataset from {len(df)} to 2000 records...")
          df = df.sample(n=2000, random_state=42)

        # Convert DataFrame to enhanced documents
        documents = self.create_enhanced_documents(df, analyzer)

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150,
            length_function=len,
            separators=["\n\n", "\n", "; ", ", ", " ", ""]
        )
        split_docs = text_splitter.split_documents(documents)

        # Create embeddings and store in Pinecone
        vectors = []
        batch_size = 25

        for i in range(0, len(split_docs), batch_size):
          batch_docs = split_docs[i:i + batch_size]
          logger.info(f"Processing embedding batch {i//batch_size + 1}/{(len(split_docs)-1)//batch_size + 1}")

          for j, doc in enumerate(batch_docs):
            embedding = self.embeddings.embed_query(doc.page_content)
            vector_id = f"kaggle_{i+j}_{hashlib.md5(doc.page_content.encode()).hexdigest()[:8]}"

            vectors.append({
                'id': vector_id,
                'values': embedding,
                'metadata': {
                    'content': doc.page_content,
                    'chunk_id': i+j,
                    **doc.metadata
                }
            })

        # Batch upsert to Pinecone
        upsert_batch_size = 100
        for i in range(0, len(vectors), upsert_batch_size):
          batch = vectors[i:i + upsert_batch_size]
          self.index.upsert(vectors=batch)

        logger.info(f"✅ Stored {len(vectors)} vectors for Kaggle dataset")

    except Exception as e:
      logger.error(f"Error processing Kaggle dataset: {str(e)}")
      raise

  #BEFORE OPTIMISATION
  def create_enhanced_documents(self, df: pd.DataFrame, analyzer: KaggleDatasetAnalyzer) -> List[Document]:
    """
    Create enhanced documents with rich medical context specific to Kaggle dataset
    """
    documents = []

    for idx, row in df.iterrows():
      # Create comprehensive patient record content
      enhanced_content = self.create_patient_record_content(row, analyzer)

      # Extract medical context
      medical_context = self.extract_medical_context(row, analyzer)

      # Create semantic enhancement
      semantic_content = self.create_semantic_enhancement(enhanced_content, medical_context)

      # Final optimized content
      final_content = f"""
      COMPREHENSIVE PATIENT HEALTHCARE RECORD:
      {enhanced_content}

      MEDICAL SEMANTIC CONTEXT:
      {semantic_content}

      CLINICAL PATTERNS AND RELATIONSHIPS:
      {self.add_pattern_context(row, analyzer)}

      HEALTHCARE DELIVERY CONTEXT:
      patient care medical treatment clinical intervention healthcare management
      hospital admission medical evaluation diagnostic assessment therapeutic intervention
      comprehensive patient record medical documentation clinical data healthcare delivery
      """.strip()

      doc = Document(
          page_content=final_content,
          metadata={
              'row_id': idx,
              'medical_condition': row['Medical Condition'],
              'age_group': self.get_age_group(row['Age']),
              'admission_type': row['Admission Type'],
              'test_result': row['Test Results'],
              'source': 'kaggle_healthcare_dataset'
          }
      )
      documents.append(doc)

    return documents

  def create_patient_record_content(self, row, analyzer):
    """Create rich patient record content"""
    return f"""
    Patient Demographics: {row['Age']} year old {row['Gender']} patient with blood type {row['Blood Type']}
    Primary Medical Condition: {row['Medical Condition']} requiring specialized medical attention
    Clinical Management: Currently treated with {row['Medication']} therapeutic intervention
    Healthcare Provider: {row['Doctor']} providing expert medical care at {row['Hospital']}
    Hospital Admission: {row['Admission Type']} admission on {row['Date of Admission']}
    Diagnostic Results: {row['Test Results']} laboratory and diagnostic findings
    Insurance Coverage: {row['Insurance Provider']} healthcare insurance plan
    Financial Information: ${row['Billing Amount']} total medical charges
    Room Assignment: Room {row['Room Number']} hospital accommodation
    Treatment Timeline: Admitted {row['Date of Admission']} discharged {row['Discharge Date']}

    MEDICAL CONDITION CONTEXT: {self.get_condition_context(row['Medical Condition'], analyzer)}
    MEDICATION THERAPY: {self.get_medication_context(row['Medication'], analyzer)}
    AGE-SPECIFIC CARE: {self.get_age_specific_context(row['Age'], row['Medical Condition'], analyzer)}
    """.strip()

  def extract_medical_context(self, row, analyzer):
    """Extract medical context based on learned patterns"""
    context_parts = []

    # Medical condition context
    condition = row['Medical Condition']
    if condition in analyzer.relationships.get('Age_MedicalCondition', {}):
      age_info = analyzer.relationships['Age_MedicalCondition'][condition]
      context_parts.append(f"Typical age demographic for {condition}: {age_info['common_age_group']}")

    # Medication context
    if condition in analyzer.relationships.get('Condition_Medication', {}):
      common_meds = list(analyzer.relationships['Condition_Medication'][condition].keys())[:3]
      context_parts.append(f"Common medications for {condition}: {', '.join(common_meds)}")

    # Test results context
    if condition in analyzer.relationships.get('Condition_TestResults', {}):
      test_patterns = analyzer.relationships['Condition_TestResults'][condition]
      most_common_result = max(test_patterns, key=test_patterns.get)
      context_parts.append(f"Most common test result for {condition}: {most_common_result}")

    return ' | '.join(context_parts)

  def create_semantic_enhancement(self, content, medical_context):
    """Create semantic enhancement for better embeddings"""
    # Medical terminology expansion
    medical_terms = [
        'healthcare delivery', 'clinical medicine', 'patient care management',
        'medical intervention', 'therapeutic outcome', 'diagnostic evaluation',
        'clinical assessment', 'medical treatment', 'healthcare services',
        'medical diagnosis', 'therapeutic intervention', 'clinical monitoring'
    ]

    # Condition-specific enhancements
    condition_enhancements = {
        'Diabetes': 'endocrine disorder glucose metabolism insulin therapy blood sugar management diabetic care',
        'Hypertension': 'cardiovascular condition blood pressure management cardiac health vascular disease',
        'Asthma': 'respiratory condition pulmonary disease breathing disorder airway management',
        'Arthritis': 'joint disorder musculoskeletal condition inflammatory disease joint pain management',
        'Cancer': 'oncological condition malignancy tumor treatment cancer therapy oncology care',
        'Obesity': 'metabolic condition weight management lifestyle intervention nutritional therapy'
    }

    # Extract condition from content
    enhanced_terms = medical_terms.copy()
    for condition, enhancement in condition_enhancements.items():
      if condition.lower() in content.lower():
        enhanced_terms.append(enhancement)

    return f"{' '.join(enhanced_terms)} {medical_context}"

  def add_pattern_context(self, row, analyzer):
    """Add learned pattern context"""
    patterns = []

    # Billing patterns
    condition = row['Medical Condition']
    if condition in analyzer.relationships.get('Billing_Condition', {}):
      billing_info = analyzer.relationships['Billing_Condition'][condition]
      patterns.append(f"Expected billing range for {condition}: ${billing_info['range'][0]:.0f}-${billing_info['range'][1]:.0f}")

    # Admission patterns
    admission_type = row['Admission Type']
    if admission_type in analyzer.relationships.get('AdmissionType_Condition', {}):
      common_conditions = list(analyzer.relationships['AdmissionType_Condition'][admission_type].keys())[:3]
      patterns.append(f"Common conditions for {admission_type} admissions: {', '.join(common_conditions)}")

    return ' | '.join(patterns)

  def get_condition_context(self, condition, analyzer):
    """Get condition-specific context"""
    context_map = {
        'Diabetes': 'chronic metabolic disorder requiring glucose monitoring insulin therapy dietary management',
        'Hypertension': 'cardiovascular condition requiring blood pressure monitoring antihypertensive medication',
        'Asthma': 'chronic respiratory condition requiring bronchodilator therapy pulmonary function monitoring',
        'Arthritis': 'inflammatory joint condition requiring pain management mobility support therapy',
        'Cancer': 'oncological condition requiring multidisciplinary treatment chemotherapy radiation therapy',
        'Obesity': 'metabolic condition requiring weight management lifestyle modification nutritional counseling'
    }
    return context_map.get(condition, 'medical condition requiring specialized healthcare management')

  def get_medication_context(self, medication, analyzer):
    """Get medication-specific context"""
    # Common medication contexts based on typical uses
    med_contexts = {
        'Aspirin': 'antiplatelet therapy cardiovascular protection pain management anti-inflammatory',
        'Ibuprofen': 'nonsteroidal anti-inflammatory pain relief fever reduction',
        'Paracetamol': 'analgesic antipyretic pain management fever control',
        'Lisinopril': 'ACE inhibitor blood pressure control cardiovascular protection',
        'Metformin': 'antidiabetic medication glucose control insulin sensitivity',
        'Penicillin': 'antibiotic therapy bacterial infection treatment',
        'Lipitor': 'statin therapy cholesterol management cardiovascular risk reduction'
    }
    return med_contexts.get(medication, 'therapeutic medication prescribed for medical condition management')

  def get_age_specific_context(self, age, condition, analyzer):
    """Get age-specific care context"""
    if age < 30:
      return 'young adult healthcare preventive care early intervention health promotion'
    elif age < 50:
      return 'middle-aged healthcare chronic disease prevention health maintenance screening'
    elif age < 70:
      return 'senior healthcare chronic disease management health monitoring wellness care'
    else:
      return 'elderly healthcare comprehensive geriatric care chronic condition management'

  def get_age_group(self, age):
    """Categorize age into groups"""
    if age < 30:
        return 'young_adult'
    elif age < 50:
        return 'middle_aged'
    elif age < 70:
        return 'senior'
    else:
        return 'elderly'

  def search_similar_patterns(self, query: str, top_k: int = 10) -> List[Dict]:
    """
    Search for similar patterns in the Kaggle dataset
    """
    try:
      # Enhanced query for Kaggle dataset
      enhanced_query = self.enhance_kaggle_query(query)

      # Get embedding and search
      query_embedding = self.embeddings.embed_query(enhanced_query)

      results = self.index.query(
          vector=query_embedding,
          top_k=top_k,
          include_metadata=True
      )

      return results['matches']

    except Exception as e:
      logger.error(f"Error searching patterns: {str(e)}")
      return []
  #BEFORE OPTIMISATION
  def enhance_kaggle_query(self, query: str) -> str:
    """
    Enhance query specifically for Kaggle healthcare dataset patterns
    """
    # Kaggle dataset specific enhancements
    kaggle_enhancements = [
        'comprehensive patient record', 'healthcare dataset', 'medical condition',
        'patient demographics', 'hospital admission', 'medical treatment',
        'diagnostic results', 'insurance coverage', 'billing information',
        'room assignment', 'medication therapy', 'test results'
    ]

    # Medical condition mappings for query enhancement
    condition_mappings = {
        'diabetes': 'diabetes mellitus glucose blood sugar endocrine disorder insulin therapy',
        'heart': 'cardiovascular cardiac hypertension blood pressure heart disease',
        'lung': 'respiratory pulmonary asthma breathing airway disease',
        'cancer': 'oncology malignancy tumor chemotherapy radiation therapy',
        'joint': 'arthritis musculoskeletal joint pain inflammatory disorder'
    }

    enhanced_query = query.lower()

    # Apply condition mappings
    for condition, enhancement in condition_mappings.items():
      if condition in enhanced_query:
        enhanced_query = enhanced_query.replace(condition, f"{condition} {enhancement}")

    # Add Kaggle-specific context
    final_query = f"""
    {' '.join(kaggle_enhancements)}
    {enhanced_query}

    healthcare delivery clinical medicine patient care management
    medical intervention therapeutic outcome diagnostic evaluation
    comprehensive medical record patient documentation healthcare information
    """.strip()

    return final_query
#-----------------------------------------------------------------------------------------------------#
#OPTIMISATION FUNCTIONS
  # def create_medical_terminology_expansion(self, content: str, condition: str) -> str:
  #   """Add medical terminology expansion for better matching"""
  #   # Medical terminology mappings
  #   medical_expansions = {
  #       'Diabetes': [
  #           'diabetes mellitus', 'hyperglycemia', 'insulin resistance', 'glucose intolerance',
  #           'diabetic ketoacidosis', 'hemoglobin A1c', 'glycemic control', 'blood glucose',
  #           'endocrine disorder', 'metabolic syndrome', 'insulin therapy', 'diabetic complications',
  #           'glucose monitoring', 'pancreatic dysfunction', 'glycosuria', 'diabetic neuropathy'
  #       ],
  #       'Hypertension': [
  #           'high blood pressure', 'elevated blood pressure', 'cardiovascular disease',
  #           'systolic pressure', 'diastolic pressure', 'hypertensive crisis', 'blood pressure control',
  #           'antihypertensive therapy', 'vascular resistance', 'cardiac output', 'arterial pressure',
  #           'hypertensive heart disease', 'essential hypertension', 'secondary hypertension'
  #       ],
  #       'Cancer': [
  #           'malignancy', 'neoplasm', 'tumor', 'carcinoma', 'oncology', 'metastasis',
  #           'chemotherapy', 'radiation therapy', 'immunotherapy', 'oncological treatment',
  #           'cancer staging', 'malignant growth', 'cancer cells', 'tumor markers',
  #           'cancer therapy', 'oncological care', 'cancer management', 'neoplastic disease'
  #       ],
  #       'Asthma': [
  #           'respiratory disease', 'airway obstruction', 'bronchospasm', 'wheezing',
  #           'dyspnea', 'shortness of breath', 'bronchodilator', 'inhaler therapy',
  #           'respiratory inflammation', 'airway hyperresponsiveness', 'allergic asthma',
  #           'asthmatic episodes', 'respiratory management', 'pulmonary function'
  #       ],
  #       'Arthritis': [
  #           'joint inflammation', 'arthritic condition', 'joint pain', 'joint stiffness',
  #           'musculoskeletal disorder', 'inflammatory arthritis', 'joint disease',
  #           'rheumatoid arthritis', 'osteoarthritis', 'joint mobility', 'arthritis management',
  #           'anti-inflammatory treatment', 'joint degeneration', 'synovial inflammation'
  #       ]
  #   }

  #   # Get expansions for the condition
  #   expansions = medical_expansions.get(condition, [])

  #   # Add general medical terms
  #   general_terms = [
  #       'clinical presentation', 'medical diagnosis', 'therapeutic intervention',
  #       'patient assessment', 'clinical management', 'medical treatment',
  #       'healthcare delivery', 'clinical care', 'medical intervention',
  #       'patient care', 'clinical evaluation', 'medical monitoring'
  #   ]

  #   # Combine all expansions
  #   all_expansions = expansions + general_terms

  #   return f"{content}\n\nMEDICAL TERMINOLOGY EXPANSION: {' '.join(all_expansions)}"

  # def add_semantic_density_boost(self, content: str, metadata: dict) -> str:
  #   """Add semantic density for better similarity matching"""
  #   condition = metadata.get('medical_condition', '')
  #   age_group = metadata.get('age_group', '')
  #   admission_type = metadata.get('admission_type', '')

  #   # Build semantic density layers
  #   density_layers = []

  #   # Layer 1: Condition-specific semantic boost
  #   if condition:
  #       condition_lower = condition.lower()
  #       if 'diabetes' in condition_lower:
  #           density_layers.append('diabetes diabetic glucose insulin blood sugar hyperglycemia endocrine metabolic syndrome glycemic diabetic diabetic')
  #       elif 'hypertension' in condition_lower:
  #           density_layers.append('hypertension blood pressure cardiovascular cardiac heart vascular antihypertensive hypertensive hypertensive')
  #       elif 'cancer' in condition_lower:
  #           density_layers.append('cancer malignancy tumor oncology neoplasm chemotherapy radiation oncological cancer cancer')
  #       elif 'asthma' in condition_lower:
  #           density_layers.append('asthma respiratory breathing airway bronchial pulmonary inhaler asthmatic asthmatic')

  #   # Layer 2: Age group semantic boost
  #   age_semantics = {
  #       'young_adult': 'young adult early adulthood preventive care health promotion',
  #       'middle_aged': 'middle aged adult chronic disease prevention health maintenance',
  #       'senior': 'senior elderly geriatric age-related chronic condition management',
  #       'elderly': 'elderly geriatric senior comprehensive care chronic disease management'
  #   }

  #   if age_group in age_semantics:
  #       density_layers.append(age_semantics[age_group])

  #   # Layer 3: Admission type boost
  #   admission_semantics = {
  #       'Emergency': 'emergency urgent critical acute immediate intervention emergency emergency',
  #       'Elective': 'elective scheduled planned routine non-urgent elective elective',
  #       'Urgent': 'urgent priority immediate attention urgent urgent'
  #   }

  #   if admission_type in admission_semantics:
  #       density_layers.append(admission_semantics[admission_type])

  #   # Layer 4: General medical semantic density
  #   general_medical_density = [
  #       'medical clinical healthcare patient treatment diagnosis therapy medication',
  #       'hospital physician doctor nurse healthcare provider medical professional',
  #       'patient care medical management clinical intervention therapeutic outcome',
  #       'medical record healthcare documentation patient information clinical data'
  #   ]

  #   density_layers.extend(general_medical_density)

  #   # Combine with original content
  #   enhanced_content = f"""
  #   {content}

  #   SEMANTIC DENSITY ENHANCEMENT:
  #   {' '.join(density_layers)}

  #   MEDICAL CONTEXT AMPLIFICATION:
  #   comprehensive patient healthcare record medical condition clinical management
  #   therapeutic intervention diagnostic evaluation patient care medical documentation
  #   """.strip()

  #   return enhanced_content

  # def create_enhanced_documents(self, df: pd.DataFrame, analyzer: KaggleDatasetAnalyzer) -> List[Document]:
  #   """Create enhanced documents with optimized semantic content"""
  #   documents = []

  #   for idx, row in df.iterrows():
  #       # Create base content
  #       base_content = self.create_patient_record_content(row, analyzer)

  #       # Add medical terminology expansion
  #       condition = row['Medical Condition']
  #       terminology_expanded = self.create_medical_terminology_expansion(base_content, condition)

  #       # Extract medical context
  #       medical_context = self.extract_medical_context(row, analyzer)

  #       # Create semantic enhancement
  #       semantic_content = self.create_semantic_enhancement(terminology_expanded, medical_context)

  #       # Add pattern context
  #       pattern_context = self.add_pattern_context(row, analyzer)

  #       # Create metadata
  #       metadata = {
  #           'row_id': idx,
  #           'medical_condition': condition,
  #           'age_group': self.get_age_group(row['Age']),
  #           'admission_type': row['Admission Type'],
  #           'test_result': row['Test Results'],
  #           'source': 'kaggle_healthcare_dataset'
  #       }

  #       # Apply semantic density boost
  #       final_content = self.add_semantic_density_boost(
  #           f"""
  #           COMPREHENSIVE PATIENT HEALTHCARE RECORD:
  #           {semantic_content}

  #           CLINICAL PATTERNS AND RELATIONSHIPS:
  #           {pattern_context}

  #           HEALTHCARE DELIVERY CONTEXT:
  #           patient care medical treatment clinical intervention healthcare management
  #           hospital admission medical evaluation diagnostic assessment therapeutic intervention
  #           comprehensive patient record medical documentation clinical data healthcare delivery
  #           """.strip(),
  #           metadata
  #       )

  #       doc = Document(
  #           page_content=final_content,
  #           metadata=metadata
  #       )
  #       documents.append(doc)

  #   return documents

  # def enhance_kaggle_query(self, query: str) -> str:
  #   """Enhanced query optimization for better similarity scores"""
  #   # Multi-layer query enhancement
  #   enhanced_layers = []

  #   # Layer 1: Direct medical condition detection and expansion
  #   query_lower = query.lower()
  #   detected_conditions = []

  #   condition_keywords = {
  #       'diabetes': ['diabetes', 'diabetic', 'insulin', 'glucose', 'blood sugar'],
  #       'hypertension': ['hypertension', 'blood pressure', 'cardiac', 'heart'],
  #       'cancer': ['cancer', 'malignancy', 'tumor', 'oncology', 'chemotherapy'],
  #       'asthma': ['asthma', 'respiratory', 'breathing', 'inhaler', 'lung'],
  #       'arthritis': ['arthritis', 'joint', 'inflammatory', 'rheumatoid']
  #   }

  #   for condition, keywords in condition_keywords.items():
  #       if any(keyword in query_lower for keyword in keywords):
  #           detected_conditions.append(condition)

  #   # Layer 2: Condition-specific expansions
  #   for condition in detected_conditions:
  #       if condition == 'diabetes':
  #           enhanced_layers.append('diabetes mellitus hyperglycemia insulin therapy glucose control endocrine disorder metabolic syndrome blood sugar management diabetic care glycemic monitoring')
  #       elif condition == 'hypertension':
  #           enhanced_layers.append('hypertension high blood pressure cardiovascular disease cardiac health vascular management antihypertensive therapy blood pressure control')
  #       elif condition == 'cancer':
  #           enhanced_layers.append('cancer malignancy neoplasm tumor oncology chemotherapy radiation therapy immunotherapy cancer treatment oncological care')
  #       elif condition == 'asthma':
  #           enhanced_layers.append('asthma respiratory disease airway obstruction bronchospasm breathing disorder inhaler therapy pulmonary function')
  #       elif condition == 'arthritis':
  #           enhanced_layers.append('arthritis joint inflammation musculoskeletal disorder joint pain inflammatory condition rheumatoid arthritis')

  #   # Layer 3: Query type enhancement
  #   if any(word in query_lower for word in ['generate', 'create', 'produce']):
  #       enhanced_layers.append('patient record generation synthetic data creation medical record synthesis healthcare data generation')

  #   if any(word in query_lower for word in ['emergency', 'urgent', 'critical']):
  #       enhanced_layers.append('emergency medical care urgent healthcare intervention critical patient management acute medical condition')

  #   if any(word in query_lower for word in ['medication', 'treatment', 'therapy']):
  #       enhanced_layers.append('pharmaceutical intervention medication therapy therapeutic treatment drug therapy medical treatment')

  #   # Layer 4: Age-specific enhancements
  #   if any(word in query_lower for word in ['aged', 'age', 'elderly', 'young']):
  #       enhanced_layers.append('age-specific healthcare demographic patient care age-related medical management')

  #   # Layer 5: Kaggle dataset structure alignment
  #   dataset_structure_terms = [
  #       'patient demographics medical condition hospital admission insurance coverage',
  #       'diagnostic test results medication prescription healthcare provider billing information',
  #       'room assignment admission type discharge planning comprehensive patient care',
  #       'blood type medical history clinical documentation healthcare record'
  #   ]

  #   # Combine all layers
  #   final_enhanced_query = f"""
  #   ORIGINAL QUERY: {query}

  #   MEDICAL CONDITION FOCUS: {' '.join(enhanced_layers)}

  #   HEALTHCARE DATASET CONTEXT: {' '.join(dataset_structure_terms)}

  #   CLINICAL SEMANTIC DENSITY:
  #   healthcare delivery clinical medicine patient care management medical intervention
  #   therapeutic outcome diagnostic evaluation clinical assessment medical treatment
  #   comprehensive medical record patient documentation healthcare information
  #   medical diagnosis clinical evaluation healthcare services medical monitoring
  #   patient assessment clinical management healthcare intervention medical care
  #   """.strip()

  #   return final_enhanced_query

# ========== CELL 27 ==========
class KaggleSyntheticDataGenerator:
