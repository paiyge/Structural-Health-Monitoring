from src.data_loader import get_data
from src.preprocessing import preprocess_data

# Simulate data
df = get_data(source='simulated')

# Or load real NASA bearing data
# df = get_data(source='nasa', file_path='data/bearing1_1.csv')


cleaned_data = preprocess_data(df, normalize=True, segment_length=None)
