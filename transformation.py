import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import gspread
import os
import time
import random
from typing import Tuple, Dict, List, Any
from datetime import datetime

SHEET_NAMES = {
    'STOCK_INFLOW': 'stock_inflow',
    'RELEASE': 'release',
    'STOCK_INFLOW_CLEAN': 'stock_inflow_clean',
    'RELEASE_CLEAN': 'release_clean',
    'SUMMARY': 'summary'
}


DATE_FORMATS = ['%d %b %Y', '%d/%m/%y', '%d-%b-%Y']
GOOGLE_SHEETS_SCOPE = ['https://www.googleapis.com/auth/spreadsheets']

class DataProcessingError(Exception):
    """Custom exception for data processing errors"""
    pass

def exponential_backoff_with_jitter(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0, jitter: bool = True) -> float:
    """Calculate exponential backoff delay with optional jitter to avoid thundering herd."""
    delay = min(base_delay * (2 ** attempt), max_delay)
    if jitter:
        delay = delay * (0.5 + random.random() * 0.5)  # Add 0-50% jitter
    return delay

def api_call_with_retry(func, max_retries: int = 5, backoff_base: float = 1.0):
    """Execute API calls with exponential backoff retry logic for rate limiting."""
    for attempt in range(max_retries + 1):
        try:
            return func()
        except (HttpError, Exception) as e:
            error_str = str(e).lower()
            
            # Check if it's a rate limit error
            is_rate_limit = any(keyword in error_str for keyword in [
                'quota exceeded', 'rate limit', '429', 'too many requests',
                'user rate limit exceeded', 'rateLimitExceeded'
            ])
            
            # Check if it's a temporary server error
            is_server_error = any(keyword in error_str for keyword in [
                '500', '502', '503', '504', 'internal error', 'backend error'
            ])
            
            if attempt == max_retries:
                print(f"API call failed after {max_retries + 1} attempts: {str(e)}")
                raise e
            
            if is_rate_limit or is_server_error:
                delay = exponential_backoff_with_jitter(attempt, backoff_base)
                print(f"API rate limit/server error detected (attempt {attempt + 1}/{max_retries + 1}). "
                      f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                continue
            else:
                # Non-retryable error, raise immediately
                raise e
    
    raise Exception(f"API call failed after {max_retries + 1} attempts")

def get_credentials(credentials_file: str) -> service_account.Credentials:
    """Create and return credentials for Google Sheets access"""
    try:
        return service_account.Credentials.from_service_account_file(
            credentials_file,
            scopes=GOOGLE_SHEETS_SCOPE
        )
    except Exception as e:
        raise DataProcessingError(f"Failed to create credentials: {str(e)}")

def connect_to_sheets(credentials: service_account.Credentials, spreadsheet_id: str) -> gspread.Spreadsheet:
    try:
        gc = gspread.authorize(credentials)
        return gc.open_by_key(spreadsheet_id)
    except Exception as e:
        raise DataProcessingError(f"Failed to connect to Google Sheets: {str(e)}")

def read_worksheet_to_df(spreadsheet: gspread.Spreadsheet, worksheet_name: str) -> pd.DataFrame:
    try:
        def _get_worksheet_data():
            worksheet = spreadsheet.worksheet(worksheet_name)
            return worksheet.get_all_values()
        
        all_values = api_call_with_retry(_get_worksheet_data, max_retries=3, backoff_base=2.0)
        
        if not all_values:
            raise DataProcessingError(f"No data found in worksheet {worksheet_name}")
        
        headers = all_values[0]
        data = all_values[1:]
        df = pd.DataFrame(data, columns=headers)
        
        if 'date' in df.columns:
            print(f"\nUnique date values in {worksheet_name}:")
        
        return df
    except Exception as e:
        raise DataProcessingError(f"Failed to read worksheet {worksheet_name}: {str(e)}")

def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    try:
        print("\nStandardizing dataframe...")
        
        df_clean = df.copy()
        
        # Standardize column names
        df_clean.columns = (df_clean.columns.str.lower()
                          .str.strip()
                          .str.replace(' ', '_')
                          .str.replace('-', '_'))
        
        # Handle the weight_in_kg to weight rename
        if 'weight_in_kg' in df_clean.columns:
            df_clean = df_clean.rename(columns={'weight_in_kg': 'weight'})
        
        for column in df_clean.columns:
            df_clean[column] = df_clean[column].astype(str).str.strip().str.lower()
            try:
                numeric_values = pd.to_numeric(df_clean[column].str.replace(',', ''))
                df_clean[column] = numeric_values
            except (ValueError, TypeError):
                pass
        
        return df_clean
    except Exception as e:
        raise DataProcessingError(f"Failed to standardize dataframe: {str(e)}")

def standardize_dates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    
    try:
        print("\nStandardizing dates...")
        df = df.copy()
        
        date_parsed = False
        for format in DATE_FORMATS:
            try:
                print(f"Trying date format: {format}")
                df['date'] = pd.to_datetime(df['date'], format=format)
                date_parsed = True
                print("Successfully parsed dates using format:", format)
                break
            except ValueError as e:
                print(f"Failed to parse with format {format}: {str(e)}")
                continue
        
        if not date_parsed:
            print("Falling back to mixed format parsing")
            df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True)
        
        if df['date'].isna().any():
            problematic_dates = df[df['date'].isna()]['date'].unique()
            print("Warning: Failed to parse these dates:", problematic_dates)
            raise DataProcessingError(f"Failed to parse dates: {problematic_dates}")
        
        df['month'] = df['date'].dt.strftime('%b').str.lower()
        df['year_month'] = df['date'].dt.strftime('%Y-%b')
        
        return df
    except Exception as e:
        raise DataProcessingError(f"Failed to standardize dates: {str(e)}")


def create_summary_df(stock_inflow_df: pd.DataFrame, release_df: pd.DataFrame) -> pd.DataFrame:
    try:
        print("\nCreating summary dataframe...")
        
        all_year_months = sorted(list(set(stock_inflow_df['year_month'].unique()) | 
                                    set(release_df['year_month'].unique())))
        
        summary_df = pd.DataFrame({'year_month': all_year_months})
        summary_df['month'] = summary_df['year_month'].str.split('-').str[1].str.lower()
        summary_df = summary_df[['month', 'year_month']]
        
        # Get unique product types dynamically from the data
        unique_product_types = stock_inflow_df['product_type'].dropna().unique()
        
        # Calculate weight loss for each product
        weight_loss_summaries = {}
        for product_type in unique_product_types:
            product_data = stock_inflow_df[
                stock_inflow_df['product_type'] == product_type
            ].copy()
            
            if not product_data.empty and 'kaduna_coldroom_weight' in product_data.columns and 'weight' in product_data.columns:
                product_data['weight_loss'] = product_data['weight'] - product_data['kaduna_coldroom_weight']
                weight_loss_summaries[f'{product_type}_weight_loss'] = product_data.groupby('year_month')['weight_loss'].sum()
        
        # Create dynamic product summaries for both inflow and release
        product_summaries = {}
        
        # Get unique product types from both inflow and release data
        # Standardize casing to match inflow format (title case)
        inflow_products = stock_inflow_df['product_type'].dropna().unique()
        
        if 'product' in release_df.columns:
            # Keep product names in lowercase
            release_products = release_df['product'].dropna().unique()
        else:
            release_products = []
        
        # Process inflow data for each product type
        for product_type in inflow_products:
            product_data = stock_inflow_df[stock_inflow_df['product_type'] == product_type]
            if not product_data.empty:
                agg_dict = {'weight': 'sum'}
                # Add quantity aggregation for products that have quantity data
                if 'quantity' in product_data.columns and product_data['quantity'].notna().any():
                    agg_dict['quantity'] = 'sum'
                
                product_summaries[f'{product_type}_inflow'] = product_data.groupby('year_month').agg(agg_dict)
        
        # Process release data for each product type
        for product_type in release_products:
            product_data = release_df[release_df['product'] == product_type]
            if not product_data.empty:
                agg_dict = {'weight': 'sum'}
                # Add quantity aggregation for products that have quantity data
                if 'quantity' in product_data.columns and product_data['quantity'].notna().any():
                    agg_dict['quantity'] = 'sum'
                
                product_summaries[f'{product_type}_release'] = product_data.groupby('year_month').agg(agg_dict)
        
        # Create dynamic summary columns for inflow and release
        summary_columns = {}
        
        # Add inflow columns for each product type
        for product_type in inflow_products:
            product_key = f'{product_type}_inflow'
            if product_key in product_summaries:
                summary_key = product_type.replace(' ', '_').lower()
                if 'quantity' in product_summaries[product_key].columns:
                    summary_columns[f'total_{summary_key}_inflow_quantity'] = (product_key, 'quantity')
                if 'weight' in product_summaries[product_key].columns:
                    summary_columns[f'total_{summary_key}_inflow_weight'] = (product_key, 'weight')
        
        # Add release columns for each product type
        for product_type in release_products:
            product_key = f'{product_type}_release'
            if product_key in product_summaries:
                summary_key = product_type.replace(' ', '_').lower()
                if 'quantity' in product_summaries[product_key].columns:
                    summary_columns[f'total_{summary_key}_release_quantity'] = (product_key, 'quantity')
                if 'weight' in product_summaries[product_key].columns:
                    summary_columns[f'total_{summary_key}_release_weight'] = (product_key, 'weight')
        
        # Add weight loss columns
        for product_type in unique_product_types:
            weight_loss_key = f'{product_type}_weight_loss'
            if weight_loss_key in weight_loss_summaries:
                summary_columns[f'total_{product_type.replace(" ", "_").lower()}_weight_loss'] = weight_loss_key
        
        for col_name, summary_key in summary_columns.items():
            if isinstance(summary_key, tuple):
                # Handle existing product summaries
                summary_dict, metric = summary_key
                if metric in product_summaries[summary_dict].columns:
                    summary_df[col_name] = summary_df['year_month'].map(
                        product_summaries[summary_dict][metric]).fillna(0)
                else:
                    summary_df[col_name] = 0
            else:
                # Handle weight loss summaries
                if summary_key in weight_loss_summaries:
                    summary_df[col_name] = summary_df['year_month'].map(
                        weight_loss_summaries[summary_key]).fillna(0)
                else:
                    summary_df[col_name] = 0

        # Sort by year_month in ascending order to process chronologically
        summary_df['sort_date'] = pd.to_datetime(summary_df['year_month'], format='%Y-%b')
        summary_df = summary_df.sort_values('sort_date')

        # Create dynamic opening stock and stock balance columns
        opening_stock_columns = []
        stock_balance_columns = []
        
        # Create opening stock and balance columns for each product type
        all_products = set(inflow_products) | set(release_products)
        for product_type in all_products:
            summary_key = product_type.replace(' ', '_').lower()
            
            # Add quantity columns if product has quantity data
            inflow_key = f'{product_type}_inflow'
            if inflow_key in product_summaries and 'quantity' in product_summaries[inflow_key].columns:
                opening_stock_columns.append(f'{summary_key}_quantity_opening_stock')
                stock_balance_columns.append(f'{summary_key}_quantity_stock_balance')
            
            # Add weight columns if product has weight data
            if (inflow_key in product_summaries and 'weight' in product_summaries[inflow_key].columns) or \
               (f'{product_type}_release' in product_summaries and 'weight' in product_summaries[f'{product_type}_release'].columns):
                opening_stock_columns.append(f'{summary_key}_weight_opening_stock')
                stock_balance_columns.append(f'{summary_key}_weight_stock_balance')
        
        # Initialize all opening stock and balance columns
        for column in opening_stock_columns + stock_balance_columns:
            summary_df[column] = 0.0

        # Calculate running balances for each month
        for i in range(len(summary_df)):
            for product_type in all_products:
                summary_key = product_type.replace(' ', '_').lower()
                
                # Handle quantity columns
                qty_opening_col = f'{summary_key}_quantity_opening_stock'
                qty_balance_col = f'{summary_key}_quantity_stock_balance'
                qty_inflow_col = f'total_{summary_key}_inflow_quantity'
                qty_release_col = f'total_{summary_key}_release_quantity'
                
                if qty_opening_col in summary_df.columns and qty_balance_col in summary_df.columns:
                    if i == 0:
                        summary_df.iloc[i, summary_df.columns.get_loc(qty_opening_col)] = 0
                    else:
                        summary_df.iloc[i, summary_df.columns.get_loc(qty_opening_col)] = \
                            summary_df.iloc[i-1, summary_df.columns.get_loc(qty_balance_col)]
                    
                    # Calculate balance
                    opening_stock = summary_df.iloc[i, summary_df.columns.get_loc(qty_opening_col)]
                    inflow = summary_df.iloc[i, summary_df.columns.get_loc(qty_inflow_col)] if qty_inflow_col in summary_df.columns else 0
                    release = summary_df.iloc[i, summary_df.columns.get_loc(qty_release_col)] if qty_release_col in summary_df.columns else 0
                    summary_df.iloc[i, summary_df.columns.get_loc(qty_balance_col)] = opening_stock + inflow - release
                
                # Handle weight columns
                wt_opening_col = f'{summary_key}_weight_opening_stock'
                wt_balance_col = f'{summary_key}_weight_stock_balance'
                wt_inflow_col = f'total_{summary_key}_inflow_weight'
                wt_release_col = f'total_{summary_key}_release_weight'
                
                if wt_opening_col in summary_df.columns and wt_balance_col in summary_df.columns:
                    if i == 0:
                        summary_df.iloc[i, summary_df.columns.get_loc(wt_opening_col)] = 0
                    else:
                        summary_df.iloc[i, summary_df.columns.get_loc(wt_opening_col)] = \
                            summary_df.iloc[i-1, summary_df.columns.get_loc(wt_balance_col)]
                    
                    # Calculate balance
                    opening_stock = summary_df.iloc[i, summary_df.columns.get_loc(wt_opening_col)]
                    inflow = summary_df.iloc[i, summary_df.columns.get_loc(wt_inflow_col)] if wt_inflow_col in summary_df.columns else 0
                    release = summary_df.iloc[i, summary_df.columns.get_loc(wt_release_col)] if wt_release_col in summary_df.columns else 0
                    summary_df.iloc[i, summary_df.columns.get_loc(wt_balance_col)] = opening_stock + inflow - release

        # Add percentage change columns for weight loss BEFORE final sorting
        # This ensures we calculate change in chronological order (oldest to newest)
        for product_type in unique_product_types:
            weight_loss_col = f'total_{product_type.replace(" ", "_").lower()}_weight_loss'
            pct_change_col = f'{product_type.replace(" ", "_").lower()}_weight_loss_pct_change'
            
            if weight_loss_col in summary_df.columns:
                # Calculate percentage change (current - previous) / previous * 100
                summary_df[pct_change_col] = summary_df[weight_loss_col].pct_change() * 100
                # Replace inf and -inf with 0 (when previous month was 0)
                summary_df[pct_change_col] = summary_df[pct_change_col].replace([float('inf'), float('-inf')], 0)
                # Fill NaN values (first month) with 0
                summary_df[pct_change_col] = summary_df[pct_change_col].fillna(0)
        
        # Sort in descending order (newest first) and clean up
        summary_df = summary_df.sort_values('sort_date', ascending=False)
        summary_df['year_month'] = summary_df['sort_date'].dt.strftime('%Y-%m')
        summary_df = summary_df.drop('sort_date', axis=1)
        
        # Format all numeric columns to 3 decimal places
        numeric_columns = summary_df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_columns:
            summary_df[col] = summary_df[col].astype(float).round(3)
        
        return summary_df
    except Exception as e:
        raise DataProcessingError(f"Failed to create summary: {str(e)}")

def prepare_df_for_upload(df: pd.DataFrame) -> pd.DataFrame:
    print("\nPreparing dataframe for upload...")
    df_copy = df.copy()
    
    date_columns = df_copy.select_dtypes(include=['datetime64']).columns
    for col in date_columns:
        df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d')
    
    for col in df_copy.columns:
        df_copy[col] = df_copy[col].fillna('')
        df_copy[col] = df_copy[col].astype(str)
        df_copy[col] = df_copy[col].replace('nan', '')
    
    return df_copy

def upload_df_to_gsheet(df: pd.DataFrame, 
                       spreadsheet_id: str, 
                       sheet_name: str, 
                       service: Any) -> bool:
    try:
        print(f"\nUploading data to sheet: {sheet_name}")
        df_to_upload = prepare_df_for_upload(df)
        
        values = [df_to_upload.columns.tolist()]
        values.extend([[str(cell) if cell is not None and cell == cell else '' 
                       for cell in row] for row in df_to_upload.values.tolist()])
        
        # Clear sheet with retry logic
        def _clear_sheet():
            return service.spreadsheets().values().clear(
                spreadsheetId=spreadsheet_id,
                range=f'{sheet_name}!A1:ZZ'
            ).execute()
        
        api_call_with_retry(_clear_sheet, max_retries=3, backoff_base=1.5)
        
        # Add small delay between clear and update to avoid rate limits
        time.sleep(0.5)
        
        # Update sheet with retry logic
        def _update_sheet():
            return service.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range=f'{sheet_name}!A1',
                valueInputOption='RAW',
                body={'values': values}
            ).execute()
        
        result = api_call_with_retry(_update_sheet, max_retries=3, backoff_base=1.5)
        
        print(f"Updated {result.get('updatedCells')} cells in {sheet_name}")
        return True
        
    except Exception as e:
        print(f"Failed to upload to {sheet_name}: {str(e)}")
        return False

def process_sheets_data(stock_inflow_df: pd.DataFrame, 
                       release_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        print("\nProcessing sheets data...")
        
        stock_inflow_df = standardize_dataframe(stock_inflow_df)
        
        # Handle the weight_at_delivery to weight rename for stock_inflow only
        if 'weight_at_delivery' in stock_inflow_df.columns:
            stock_inflow_df = stock_inflow_df.rename(columns={'weight_at_delivery': 'weight'})
        
        release_df = standardize_dataframe(release_df)
        
        # Filter out rows where both date and product_type are empty for stock_inflow
        stock_inflow_df = stock_inflow_df[
            ~((stock_inflow_df['date'].isna() | (stock_inflow_df['date'] == '')) & 
              (stock_inflow_df['product_type'].isna() | (stock_inflow_df['product_type'] == '')))
        ]
        
        # Filter out rows where both date and product are empty for release
        release_df = release_df[
            ~((release_df['date'].isna() | (release_df['date'] == '')) & 
              (release_df['product'].isna() | (release_df['product'] == '')))
        ]
        
        # Check for rows with missing dates in remaining data for stock_inflow
        missing_dates_inflow = stock_inflow_df[stock_inflow_df['date'].isna() | (stock_inflow_df['date'] == '')]
        missing_dates_release = release_df[release_df['date'].isna() | (release_df['date'] == '')]
        
        if not missing_dates_inflow.empty:
            raise DataProcessingError(f"Stock inflow sheet contains {len(missing_dates_inflow)} rows with missing dates but other data present. All rows with data must have valid dates.")
        
        if not missing_dates_release.empty:
            raise DataProcessingError(f"Release sheet contains {len(missing_dates_release)} rows with missing dates. All rows must have valid dates.")
        
        stock_inflow_df = standardize_dates(stock_inflow_df)
        release_df = standardize_dates(release_df)
        
        # Set gizzard quantity to 0 in release data (gizzard is sold by weight, not quantity)
        if 'product' in release_df.columns and 'quantity' in release_df.columns:
            release_df.loc[
                release_df['product'].str.contains('gizzard', 
                                                 case=False, na=False), 
                'quantity'
            ] = 0
        
        summary_df = create_summary_df(stock_inflow_df, release_df)
        
        return stock_inflow_df, release_df, summary_df
    
    except Exception as e:
        raise DataProcessingError(f"Failed to process sheets data: {str(e)}")

def main():
    CREDENTIALS_FILE = 'service-account.json'
    
    try:
        print("\nStarting data processing...")
        
        source_spreadsheet_id = os.getenv('INVENTORY_SHEET_ID')
        output_spreadsheet_id = os.getenv('INVENTORY_ETL_SPREADSHEET_ID')
        
        if not source_spreadsheet_id:
            raise DataProcessingError("INVENTORY_SHEET_ID environment variable not set")
        if not output_spreadsheet_id:
            raise DataProcessingError("INVENTORY_ETL_SPREADSHEET_ID environment variable not set")
            
        # Create credentials and services once
        credentials = get_credentials(CREDENTIALS_FILE)
        source_spreadsheet = connect_to_sheets(credentials, source_spreadsheet_id)
        sheets_service = build('sheets', 'v4', credentials=credentials)
        
        # Read the worksheets from source
        stock_inflow_df = read_worksheet_to_df(source_spreadsheet, SHEET_NAMES['STOCK_INFLOW'])
        release_df = read_worksheet_to_df(source_spreadsheet, SHEET_NAMES['RELEASE'])
        
        # Process the data
        stock_inflow_df, release_df, summary_df = process_sheets_data(
            stock_inflow_df, release_df)
        
        # Define upload tasks
        upload_tasks = [
            (stock_inflow_df, SHEET_NAMES['STOCK_INFLOW_CLEAN']),
            (release_df, SHEET_NAMES['RELEASE_CLEAN']),
            (summary_df, SHEET_NAMES['SUMMARY'])
        ]
        
        # Upload all datasets with rate limiting between uploads
        success = True
        for i, (df, sheet_name) in enumerate(upload_tasks):
            if i > 0:  # Add delay between uploads (except for the first one)
                time.sleep(1.0)  # 1 second delay between uploads
            
            if not upload_df_to_gsheet(df, output_spreadsheet_id, sheet_name, sheets_service):
                success = False
                print(f"Failed to upload {sheet_name}")
        
        if success:
            print("\nData processing and upload completed successfully!")
        else:
            raise DataProcessingError("Failed to upload one or more datasets")
            
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()