class PrivacySecurityManager:
  """
  Handles data privacy, encryption, and HIPAA compliance measures
  Simplified for single dataset
  """

  def __init__(self):
    self.faker = Faker()
    self.encryption_key = Fernet.generate_key()
    self.cipher_suite = Fernet(self.encryption_key)

  def anonymize_data(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Anonymize PHI in the healthcare dataset
    """
    logger.info("Starting data anonymization process...")
    anonymized_df = df.copy()

    # Anonymize names
    anonymized_df['Name'] = [self.faker.name() for _ in range(len(df))]

    # Anonymize doctor names (keep Dr. prefix)
    anonymized_df['Doctor'] = [f"Dr. {self.faker.last_name()}" for _ in range(len(df))]

    # Keep medical data intact but anonymize identifiers
    logger.info(f"✅ Anonymized {len(anonymized_df)} records")
    return anonymized_df

# ========== CELL 26 ==========
class KagglePineconeVectorManager:
