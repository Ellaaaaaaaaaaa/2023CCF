import pandas as pd

def process_csv(input_file, output_file):
    df = pd.read_csv(input_file, delimiter='\t')  # Assuming your file is tab-delimited

    # Remove leading whitespaces from column names
    df.columns = df.columns.str.strip()
    # Update values in "consumption_level" and "activity_level" to a minimum of 40
    df['consumption_level'] = df['consumption_level'].apply(lambda x: max(x, 50))
    df['activity_level'] = df['activity_level'].apply(lambda x: max(x, 62))

    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_file, sep='\t', index=False)

# Example usage:
input_filename = 'predictions_BILSTM_300_0.005_64_32_FLASE_loss57.pth.csv'
output_filename = 'predictions_BILSTM_300_0.005_64_32_FLASE_loss57.pth.csv'
process_csv(input_filename, output_filename)
