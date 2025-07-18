import os
import json
import pickle
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import requests
from datetime import datetime
import pytz

# Constants for Stock Balance
SPECIFICATION_SHEET_ID = os.environ.get('SPECIFICATION_SHEET_ID')
INVENTORY_SHEET_ID = os.environ.get('INVENTORY_SHEET_ID')
if not SPECIFICATION_SHEET_ID:
    raise ValueError("SPECIFICATION_SHEET_ID environment variable not set")

if not INVENTORY_SHEET_ID:
    raise ValueError("INVENTORY_SHEET_ID environment variable not set")
    
STOCK_SHEET_NAME = 'balance'
STOCK_RANGE = 'A1:P3'  # Range covers A-P columns (Specification through TOTAL including Gizzard)

INVENTORY_SHEET_NAME = 'summary'  # The sheet name from the inventory tracking spreadsheet
INVENTORY_RANGE = 'A:ZZ'  # Get all columns dynamically (covers up to 702 columns)

PARTS_SHEET_NAME = 'parts_balance'
PARTS_RANGE = 'A1:H3'  # Adjust range to cover all parts data

SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
SERVICE_ACCOUNT_FILE = os.environ.get('SERVICE_ACCOUNT_FILE', 'service-account.json')

# Set up data directory for state persistence
DATA_DIR = os.getenv('GITHUB_WORKSPACE', os.getcwd())
os.makedirs(DATA_DIR, exist_ok=True)

# Separate state files for stock and parts
STOCK_STATE_FILE = os.path.join(DATA_DIR, 'previous_stock_state.pickle')
PARTS_STATE_FILE = os.path.join(DATA_DIR, 'previous_parts_state.pickle')

# State files for tracking discrepancies
CHICKEN_DISCREPANCY_STATE_FILE = os.path.join(DATA_DIR, 'previous_chicken_discrepancy_state.pickle')
GIZZARD_DISCREPANCY_STATE_FILE = os.path.join(DATA_DIR, 'previous_gizzard_discrepancy_state.pickle')

class APIError(Exception):
    """Custom exception for API related errors."""
    pass

def get_service():
    """Create and return Google Sheets service object."""
    try:
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        return build('sheets', 'v4', credentials=credentials)
    except Exception as e:
        print(f"Error initializing Google Sheets service: {str(e)}")
        raise APIError("Failed to initialize Google Sheets service")

def get_sheet_data(service, sheet_name, range_name):
    """Fetch data from Google Sheet."""
    print(f"Fetching data from sheet {sheet_name}...")
    try:
        sheet = service.spreadsheets()
        result = sheet.values().get(
            spreadsheetId=SPECIFICATION_SHEET_ID,
            range=f'{sheet_name}!{range_name}'
        ).execute()
        data = result.get('values', [])
        
        # Validate data structure
        min_rows = 2  # Both stock and parts sheets now have 2 rows
        if not data or len(data) < min_rows:
            raise APIError(f"Invalid data structure received from Google Sheets for {sheet_name}")
            
        print(f"Data fetched successfully from {sheet_name}")
        return data
    except HttpError as e:
        print(f"Google Sheets API error: {str(e)}")
        raise APIError(f"Failed to fetch data from Google Sheets for {sheet_name}")
    except Exception as e:
        print(f"Unexpected error fetching sheet data: {str(e)}")
        raise APIError(f"Unexpected error while fetching data from {sheet_name}")

def load_previous_state(state_file):
    """Load previous state from file."""
    print(f"Checking for previous state file {state_file}")
    try:
        if os.path.exists(state_file):
            print(f"Loading previous state from {state_file}")
            with open(state_file, 'rb') as f:
                data = pickle.load(f)
                min_rows = 2  # Both state files now expect 2 rows
                if not data or len(data) < min_rows:
                    print("Invalid state data found, treating as no previous state")
                    return None
                print("Previous state loaded successfully")
                return data
        print("No previous state file found")
        return None
    except Exception as e:
        print(f"Error loading previous state: {str(e)}")
        return None

def save_current_state(state, state_file):
    """Save current state to file."""
    min_rows = 2  # Both state files now expect 2 rows
    if not state or len(state) < min_rows:
        print("Invalid state data, skipping save")
        return
        
    print(f"Saving current state to {state_file}")
    try:
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        with open(state_file, 'wb') as f:
            pickle.dump(state, f)
        print(f"State saved successfully to {state_file}")
    except Exception as e:
        print(f"Error saving state: {str(e)}")
        raise APIError("Failed to save state file")

def detect_stock_changes(previous_data, current_data):
    """Detect changes between previous and current stock data."""
    if not previous_data:
        print("No previous stock data available")
        return []
    
    try:
        changes = []
        # Skip header row and compare the balance row
        prev_row = previous_data[1]
        curr_row = current_data[1]
        headers = current_data[0]
        
        # Validate data lengths
        if len(prev_row) != len(curr_row) or len(headers) != len(curr_row):
            print(f"Data length mismatch - Previous: {len(prev_row)}, Current: {len(curr_row)}, Headers: {len(headers)}")
            print("Resetting previous stock state file to match new structure.")
            save_current_state(current_data, STOCK_STATE_FILE)
            return []
        
        print("\nComparing stock states...")
        
        # Compare each value and convert to same type before comparison
        for i in range(len(prev_row)):
            # Convert both values to strings for comparison to avoid type mismatches
            prev_val = str(prev_row[i]).strip()
            curr_val = str(curr_row[i]).strip()
            
            if prev_val != curr_val:
                changes.append((headers[i], prev_row[i], curr_row[i]))
                print(f"Change detected in {headers[i]}")
        
        if changes:
            print(f"Detected {len(changes)} stock changes")
        else:
            print("No changes detected in stock balance")
        return changes
    except Exception as e:
        print(f"Error detecting stock changes: {str(e)}")
        raise APIError("Failed to compare stock states")

def detect_parts_changes(previous_data, current_data):
    """Detect changes between previous and current parts data."""
    if not previous_data:
        print("No previous parts data available")
        return []
    
    try:
        changes = []
        # Get part headers from row 1 (starting from column B which is index 1)
        part_headers = []
        if len(current_data) > 0 and len(current_data[0]) > 1:
            part_headers = current_data[0][1:]  # Skip "Parts Type" column
        
        # Get previous values from row 2 (starting from column B which is index 1)
        prev_values = []
        if len(previous_data) > 1 and len(previous_data[1]) > 1:
            prev_values = previous_data[1][1:]  # Skip "Balance" label
        
        # Get current values from row 2 (starting from column B which is index 1)
        curr_values = []
        if len(current_data) > 1 and len(current_data[1]) > 1:
            curr_values = current_data[1][1:]  # Skip "Balance" label
        
        # Validate data structure
        if len(part_headers) != len(curr_values):
            print(f"Warning: Mismatch between parts ({len(part_headers)}) and values ({len(curr_values)})")
            # Use the shorter length for comparison
            compare_length = min(len(part_headers), len(curr_values))
            # Trim the arrays to the same length
            part_headers = part_headers[:compare_length]
            curr_values = curr_values[:compare_length]
            prev_values = prev_values[:compare_length] if len(prev_values) > compare_length else prev_values
        
        # If previous values array is shorter than current, pad it
        if len(prev_values) < len(curr_values):
            print(f"Warning: Previous values array ({len(prev_values)}) shorter than current ({len(curr_values)})")
            # Pad with empty strings
            prev_values = prev_values + [''] * (len(curr_values) - len(prev_values))
        # If previous values array is longer, trim it
        elif len(prev_values) > len(curr_values):
            print(f"Warning: Previous values array ({len(prev_values)}) longer than current ({len(curr_values)})")
            prev_values = prev_values[:len(curr_values)]
            
        print("\nComparing parts states...")
        
        # Compare each value and detect changes
        for i in range(len(part_headers)):
            if i >= len(prev_values) or i >= len(curr_values):
                print(f"Warning: Index {i} out of bounds. Skipping comparison.")
                continue
                
            # Convert both values to strings for comparison to avoid type mismatches
            prev_val = str(prev_values[i]).strip()
            curr_val = str(curr_values[i]).strip()
            
            if prev_val != curr_val:
                changes.append((part_headers[i], prev_values[i], curr_values[i]))
                print(f"Change detected in {part_headers[i]}")
        
        # Total is now included in the part headers and values, so no separate check needed
        
        if changes:
            print(f"Detected {len(changes)} parts changes")
        else:
            print("No changes detected in parts weights")
        return changes
    except Exception as e:
        print(f"Error detecting parts changes: {str(e)}")
        print("Attempting to reset parts state file for next run...")
        # Save current state to recover from this error
        save_current_state(current_data, PARTS_STATE_FILE)
        print("Parts state file updated with current data. Next run should work correctly.")
        # Return empty changes to avoid further errors
        return []

def load_previous_discrepancy_state(state_file):
    """Load previous discrepancy state from file."""
    print(f"Checking for previous discrepancy state file {state_file}")
    try:
        if os.path.exists(state_file):
            print(f"Loading previous discrepancy state from {state_file}")
            with open(state_file, 'rb') as f:
                data = pickle.load(f)
                print(f"Previous discrepancy state loaded: {data}")
                return data
        print("No previous discrepancy state file found")
        return None
    except Exception as e:
        print(f"Error loading previous discrepancy state: {str(e)}")
        return None

def save_discrepancy_state(discrepancy_value, state_file):
    """Save discrepancy state to file."""
    print(f"Saving discrepancy state {discrepancy_value} to {state_file}")
    try:
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        with open(state_file, 'wb') as f:
            pickle.dump(discrepancy_value, f)
        print(f"Discrepancy state saved successfully to {state_file}")
    except Exception as e:
        print(f"Error saving discrepancy state: {str(e)}")
        raise APIError("Failed to save discrepancy state file")

def detect_discrepancy_changes(previous_discrepancy, current_discrepancy, product_name):
    """Detect changes in discrepancy values."""
    if previous_discrepancy is None:
        if current_discrepancy is not None and current_discrepancy != 0:
            print(f"New {product_name} discrepancy detected: {current_discrepancy}")
            return True
        return False
    
    if current_discrepancy is None:
        if previous_discrepancy != 0:
            print(f"{product_name} discrepancy resolved (was {previous_discrepancy}, now no data)")
            return True
        return False
    
    if previous_discrepancy != current_discrepancy:
        print(f"{product_name} discrepancy changed from {previous_discrepancy} to {current_discrepancy}")
        return True
    
    print(f"No change in {product_name} discrepancy: {current_discrepancy}")
    return False

def get_inventory_data(service):
    """Fetch all inventory data from the inflow/release sheet."""
    try:
        result = service.spreadsheets().values().get(
            spreadsheetId=INVENTORY_SHEET_ID,
            range=f'{INVENTORY_SHEET_NAME}!{INVENTORY_RANGE}'
        ).execute()
        
        data = result.get('values', [])
        if not data:
            print("No data found in inventory sheet")
            return None
            
        # Get the header row to find the column indices
        if len(data) < 2:  # Need at least header row and one data row
            print("Not enough rows in inventory sheet")
            return None
            
        headers = data[0]
        
        # Define all required columns
        required_columns = [
            'whole_chicken_quantity_stock_balance',
            'gizzard_weight_stock_balance',
            'year_month',
            'total_whole_chicken_weight_loss',
            'total_gizzard_weight_loss',
            'total_laps_weight_loss',
            'total_fillet_weight_loss',
            'total_bone_weight_loss',
            'total_wings_weight_loss',
            'total_breast_weight_loss',
            'whole_chicken_weight_loss_pct_change',
            'gizzard_weight_loss_pct_change',
            'laps_weight_loss_pct_change',
            'fillet_weight_loss_pct_change',
            'bone_weight_loss_pct_change',
            'wings_weight_loss_pct_change',
            'breast_weight_loss_pct_change'
        ]
        
        # Find column indices (handle missing columns gracefully)
        column_indices = {}
        
        for col in required_columns:
            try:
                column_indices[col] = headers.index(col)
            except ValueError:
                column_indices[col] = -1  # Mark as missing
            
        # Get current year-month in YYYY-MM format
        current_date = datetime.now(pytz.UTC).astimezone(pytz.timezone('Africa/Lagos'))
        current_year_month = current_date.strftime('%Y-%m')
        
        # Find the row for the current month
        data_rows = data[1:]  # Skip header row
        current_month_row = None
        
        for row in data_rows:
            if len(row) > column_indices['year_month'] and row[column_indices['year_month']] == current_year_month:
                current_month_row = row
                break
        
        if not current_month_row:
            print(f"Warning: No data found for current month ({current_year_month})")
            # Sort by year_month in descending order to get the most recent record as fallback
            sorted_data = sorted(data_rows, 
                               key=lambda x: x[column_indices['year_month']] if len(x) > column_indices['year_month'] else '', 
                               reverse=True)
            if sorted_data:
                current_month_row = sorted_data[0]
                print(f"Using most recent available data from {current_month_row[column_indices['year_month']]}")
            else:
                return None
        
        # Extract all values into a dictionary
        inventory_data = {}
        for col, index in column_indices.items():
            if index == -1:  # Column doesn't exist
                inventory_data[col] = current_year_month if col == 'year_month' else None
            elif len(current_month_row) > index:
                try:
                    if col == 'year_month':
                        inventory_data[col] = current_month_row[index]
                    else:
                        inventory_data[col] = float(current_month_row[index]) if current_month_row[index] else 0.0
                except (ValueError, TypeError):
                    print(f"Invalid value for {col} in inventory sheet")
                    inventory_data[col] = None
            else:
                inventory_data[col] = None
        
        return inventory_data
    except Exception as e:
        print(f"Error fetching inventory data: {str(e)}")
        return None

def get_inventory_balance(service):
    """Fetch chicken quantity balance from inventory data (for backward compatibility)."""
    inventory_data = get_inventory_data(service)
    if inventory_data:
        return inventory_data.get('whole_chicken_quantity_stock_balance', None)
    return None

def get_gizzard_inventory_balance(service):
    """Fetch gizzard weight balance from inventory data (for backward compatibility)."""
    inventory_data = get_inventory_data(service)
    if inventory_data:
        return inventory_data.get('gizzard_weight_stock_balance', None)
    return None

def interpret_weight_loss_pct_change(pct_change, product_name):
    """Interpret percentage change for weight loss based on professional guidelines."""
    if pct_change == 0:
        return f"{product_name} weight loss remained unchanged"
    
    abs_change = abs(pct_change)
    
    if pct_change > 0:
        # Positive percentage: generally indicates worsening (more loss or less gain)
        return f"{product_name} weight loss worsened by {abs_change:.1f}%"
    else:
        # Negative percentage: generally indicates improvement (less loss or more gain)
        return f"{product_name} weight loss improved by {abs_change:.1f}%"

def format_weight_loss_section(inventory_data):
    """Format the weight loss section of the alert message."""
    if not inventory_data:
        return ""
    
    section = "*📊 Monthly Weight Loss Analysis (Cold Room Kaduna → Nasarawa):*\n"
    
    # Weight loss totals
    weight_loss_products = [
        ('whole_chicken', 'total_whole_chicken_weight_loss'),
        ('gizzard', 'total_gizzard_weight_loss'),
        ('laps', 'total_laps_weight_loss'),
        ('fillet', 'total_fillet_weight_loss'),
        ('bone', 'total_bone_weight_loss'),
        ('wings', 'total_wings_weight_loss'),
        ('breast', 'total_breast_weight_loss')
    ]
    
    section += "\n*Current Month Weight Loss (Kaduna → Nasarawa):*\n"
    for product, loss_key in weight_loss_products:
        loss_value = inventory_data.get(loss_key, None)
        if loss_value is None:
            section += f"• {product.replace('_', ' ').title()}: Not Available\n"
        elif loss_value != 0:
            if loss_value > 0:
                section += f"• {product.replace('_', ' ').title()}: +{loss_value:.2f} kg (gain)\n"
            else:
                section += f"• {product.replace('_', ' ').title()}: {loss_value:.2f} kg (loss)\n"
        else:
            section += f"• {product.replace('_', ' ').title()}: 0.00 kg (no change)\n"
    
    # Percentage changes
    pct_change_products = [
        ('whole_chicken', 'whole_chicken_weight_loss_pct_change'),
        ('gizzard', 'gizzard_weight_loss_pct_change'),
        ('laps', 'laps_weight_loss_pct_change'),
        ('fillet', 'fillet_weight_loss_pct_change'),
        ('bone', 'bone_weight_loss_pct_change'),
        ('wings', 'wings_weight_loss_pct_change'),
        ('breast', 'breast_weight_loss_pct_change')
    ]
    
    section += "\n*Month-over-Month Performance (Current vs Previous | Kaduna → Nasarawa):*\n"
    for product, pct_key in pct_change_products:
        pct_value = inventory_data.get(pct_key, None)
        product_display = product.replace('_', ' ').title()
        
        if pct_value is None:
            section += f"• ❓ {product_display} weight loss: Not Available\n"
        else:
            interpretation = interpret_weight_loss_pct_change(pct_value, product_display)
            
            if pct_value > 0:
                section += f"• ⚠️ {interpretation}\n"
            elif pct_value < 0:
                section += f"• ✅ {interpretation}\n"
            else:
                section += f"• ➖ {interpretation}\n"
    
    return section + "\n"

def calculate_total_pieces(stock_data):
    """Calculate total pieces from stock data, excluding Gizzard."""
    try:
        headers = stock_data[0]
        values = stock_data[1]
        total = 0
        
        for i in range(len(headers)):
            header = headers[i].lower()
            if header != 'specification' and header != 'gizzard' and header != 'total':
                try:
                    val = values[i]
                    if str(val).strip().replace(',', '').isdigit():
                        total += int(float(val))
                except (ValueError, TypeError):
                    continue
        return total
    except Exception as e:
        print(f"Error calculating total pieces: {str(e)}")
        return None

def calculate_chicken_discrepancy(stock_data, inventory_data):
    """Calculate discrepancy between specification sheet and inventory for whole chicken."""
    if not inventory_data:
        return None
        
    try:
        total_pieces = calculate_total_pieces(stock_data)
        inventory_balance = inventory_data.get('whole_chicken_quantity_stock_balance')
        
        if total_pieces is None or inventory_balance is None:
            return None
            
        # Return the difference (specification - inventory)
        return int(total_pieces - inventory_balance)
    except Exception as e:
        print(f"Error calculating chicken discrepancy: {str(e)}")
        return None

def calculate_gizzard_discrepancy(stock_data, inventory_data):
    """Calculate discrepancy between specification sheet and inventory for gizzard."""
    if not inventory_data:
        return None
        
    try:
        headers = stock_data[0]
        values = stock_data[1]
        gizzard_weight = 0
        
        # Find gizzard weight in specification sheet
        for i in range(len(headers)):
            if headers[i].lower() == 'gizzard':
                try:
                    gizzard_weight = float(values[i])
                    break
                except (ValueError, TypeError):
                    continue
        
        gizzard_inventory_balance = inventory_data.get('gizzard_weight_stock_balance')
        
        if gizzard_weight == 0 or gizzard_inventory_balance is None:
            return None
            
        # Return the difference (specification - inventory)
        return round(gizzard_weight - gizzard_inventory_balance, 2)
    except Exception as e:
        print(f"Error calculating gizzard discrepancy: {str(e)}")
        return None

def format_stock_section(stock_changes, stock_data, inventory_data=None):
    """Format the stock section of the alert message."""
    section = ""
    
    # Add stock changes if any
    if stock_changes:
        section += "*Stock Balance Changes:*\n"
        for spec, old_val, new_val in stock_changes:
            # Capitalize first letter of specification
            spec = spec.title()
            
            # Check if this is a weight-based value (like Gizzard)
            is_weight = spec.lower() == "gizzard"
            
            # Try to convert values to numbers and append appropriate units
            try:
                if is_weight:
                    # Handle weight values (in kg)
                    old_val_num = float(old_val) if str(old_val).strip().replace('.', '', 1).isdigit() else None
                    new_val_num = float(new_val) if str(new_val).strip().replace('.', '', 1).isdigit() else None
                    
                    if old_val_num is not None:
                        old_val_str = f"{old_val_num:,.2f} kg"
                    else:
                        old_val_str = str(old_val)
                        
                    if new_val_num is not None:
                        new_val_str = f"{new_val_num:,.2f} kg"
                    else:
                        new_val_str = str(new_val)
                else:
                    # Handle piece-based values
                    old_val_num = float(old_val) if str(old_val).strip().replace(',', '').isdigit() else None
                    new_val_num = float(new_val) if str(new_val).strip().replace(',', '').isdigit() else None
                    
                    if old_val_num is not None:
                        old_suffix = " piece" if old_val_num == 1 else " pieces"
                        old_val_str = f"{old_val_num:,.0f}{old_suffix}"
                    else:
                        old_val_str = str(old_val)
                        
                    if new_val_num is not None:
                        new_suffix = " piece" if new_val_num == 1 else " pieces"
                        new_val_str = f"{new_val_num:,.0f}{new_suffix}"
                    else:
                        new_val_str = str(new_val)
                
                section += f"• {spec}: {old_val_str} → {new_val_str}\n"
            except (ValueError, TypeError):
                section += f"• {spec}: {old_val} → {new_val}\n"
        section += "\n"
    
    # Always add current stock levels
    section += "*Current Stock Levels:*\n"
    headers = stock_data[0]
    values = stock_data[1]
    total_pieces = 0
    current_gizzard_weight = 0
    
    for i in range(len(headers)):
        # Skip 'Specification' header if it exists
        if headers[i].lower() != 'specification':
            try:
                # Capitalize first letter of header
                header = headers[i].title()
                
                # Check if this is a weight-based value (like Gizzard)
                is_weight = header.lower() == "gizzard"
                
                # Try to convert value to number and format appropriately
                val = values[i]
                if is_weight:
                    # Handle weight values (in kg)
                    if str(val).strip().replace('.', '', 1).isdigit():
                        current_gizzard_weight = float(val)
                        formatted_val = f"{current_gizzard_weight:,.2f} kg"
                    else:
                        formatted_val = str(val)
                else:
                    # Handle piece-based values
                    if str(val).strip().replace(',', '').isdigit():
                        total_pieces = int(float(val)) if header.lower() == 'total' else total_pieces
                        total_val = int(float(val))
                        bags = total_val // 20
                        remaining_pieces = total_val % 20
                        
                        # Use proper singular/plural forms
                        bags_text = "1 bag" if bags == 1 else f"{bags:,} bags"
                        pieces_text = "1 piece" if remaining_pieces == 1 else f"{remaining_pieces} pieces"
                        
                        if bags > 0 and remaining_pieces > 0:
                            formatted_val = f"{bags_text}, {pieces_text}"
                        elif bags > 0:
                            formatted_val = bags_text
                        else:
                            formatted_val = pieces_text
                    else:
                        formatted_val = str(val)
                section += f"• {header}: {formatted_val}\n"
            except (ValueError, TypeError):
                section += f"• {headers[i].title()}: {values[i]}\n"
    
    # Add inventory balance comparison if available
    if inventory_data:
        inventory_balance = inventory_data.get('whole_chicken_quantity_stock_balance')
        gizzard_inventory_balance = inventory_data.get('gizzard_weight_stock_balance')
        
        if total_pieces > 0:
            section += "\n*Whole Chicken Stock Balance Comparison:*\n"
            if inventory_balance is None:
                section += "❓ Inventory Records Total: Not Available\n"
                section += f"• Specification Sheet Total: {total_pieces:,} pieces\n"
            else:
                difference = int(total_pieces - inventory_balance)  # Convert to integer
                if difference == 0:
                    section += "✅ Chicken stock balance matches inventory records\n"
                else:
                    section += f"⚠️ Whole chicken stock balance discrepancy detected:\n"
                    section += f"• Specification Sheet Total: {total_pieces:,} pieces\n"
                    section += f"• Inventory Records Total: {int(inventory_balance):,} pieces\n"  # Convert to integer
                    section += f"• Difference: {abs(difference):,} pieces {'more' if difference > 0 else 'less'} in specification sheet\n"
        
        # Add gizzard inventory balance comparison if available
        if current_gizzard_weight > 0:
            section += "\n*Gizzard Stock Balance Comparison:*\n"
            if gizzard_inventory_balance is None:
                section += "❓ Inventory Records Gizzard: Not Available\n"
                section += f"• Specification Sheet Gizzard: {current_gizzard_weight:,.2f} kg\n"
            else:
                difference = current_gizzard_weight - gizzard_inventory_balance
                if abs(difference) < 0.01:  # Allow for small floating point differences
                    section += "✅ Gizzard stock balance matches inventory records\n"
                else:
                    section += f"⚠️ Gizzard stock balance discrepancy detected:\n"
                    section += f"• Specification Sheet Gizzard: {current_gizzard_weight:,.2f} kg\n"
                    section += f"• Inventory Records Gizzard: {gizzard_inventory_balance:,.2f} kg\n"
                    section += f"• Difference: {abs(difference):,.2f} kg {'more' if difference > 0 else 'less'} in specification sheet\n"
    
    return section

def format_parts_section(parts_changes, parts_data):
    """Format the parts section of the alert message."""
    section = ""
    
    # Add parts changes if any
    if parts_changes:
        section += "*Parts Weight Changes:*\n"
        for part, old_val, new_val in parts_changes:
            # Capitalize first letter of part name
            part = part.title()
            
            # Try to convert values to numbers with weight suffix
            try:
                # Check if values are numeric
                if str(old_val).strip().replace('.', '', 1).isdigit():
                    old_val_num = float(old_val)
                    # Use "kg" for all weights as it's a unit, not a count
                    old_val_str = f"{old_val_num:,.2f} kg"
                else:
                    old_val_str = str(old_val)
                    
                if str(new_val).strip().replace('.', '', 1).isdigit():
                    new_val_num = float(new_val)
                    new_val_str = f"{new_val_num:,.2f} kg"
                else:
                    new_val_str = str(new_val)
                    
                section += f"• {part}: {old_val_str} → {new_val_str}\n"
            except (ValueError, TypeError):
                section += f"• {part}: {old_val} → {new_val}\n"
        section += "\n"
    
    # Always add current parts weights
    section += "*Current Parts Weights:*\n"
    
    # Get part headers from row 1 (starting from column B which is index 1)
    part_headers = []
    if len(parts_data) > 0 and len(parts_data[0]) > 1:
        part_headers = parts_data[0][1:]  # Skip "Parts Type" column
    
    # Get values from row 2 (starting from column B which is index 1)
    values = []
    if len(parts_data) > 1 and len(parts_data[1]) > 1:
        values = parts_data[1][1:]  # Skip "Balance" label in row 2
    
    # Map values to headers
    for i in range(min(len(part_headers), len(values))):
        try:
            # Format weight values
            val = values[i]
            # Capitalize part name
            part_name = part_headers[i].title()
            
            if str(val).strip().replace('.', '', 1).isdigit():
                # "kg" is always singular as it's a unit
                formatted_val = f"{float(val):,.2f} kg"
            else:
                formatted_val = str(val)
            section += f"• {part_name}: {formatted_val}\n"
        except (ValueError, TypeError, IndexError) as e:
            print(f"Error formatting part {i}: {str(e)}")
            # Ensure part name is capitalized even in error case
            part_name = part_headers[i].title() if i < len(part_headers) else 'Unknown'
            section += f"• {part_name}: {values[i] if i < len(values) else 'N/A'}\n"
    
    return section

def format_discrepancy_alert(chicken_discrepancy, gizzard_discrepancy, stock_data, inventory_data):
    """Format discrepancy alert message."""
    message = "⚠️ *Nasarawa Stock Discrepancy Alert*\n\n"
    
    # Add chicken discrepancy section
    if chicken_discrepancy is not None and chicken_discrepancy != 0:
        total_pieces = calculate_total_pieces(stock_data)
        inventory_balance = inventory_data.get('whole_chicken_quantity_stock_balance') if inventory_data else None
        
        message += "*🐔 Whole Chicken Stock Discrepancy:*\n"
        if total_pieces is not None and inventory_balance is not None:
            message += f"• Specification Sheet Total: {total_pieces:,} pieces\n"
            message += f"• Inventory Records Total: {int(inventory_balance):,} pieces\n"
            message += f"• Discrepancy: {abs(chicken_discrepancy):,} pieces {'more' if chicken_discrepancy > 0 else 'less'} in specification sheet\n\n"
    
    # Add gizzard discrepancy section
    if gizzard_discrepancy is not None and gizzard_discrepancy != 0:
        headers = stock_data[0]
        values = stock_data[1]
        gizzard_weight = 0
        
        # Find gizzard weight in specification sheet
        for i in range(len(headers)):
            if headers[i].lower() == 'gizzard':
                try:
                    gizzard_weight = float(values[i])
                    break
                except (ValueError, TypeError):
                    continue
        
        gizzard_inventory_balance = inventory_data.get('gizzard_weight_stock_balance') if inventory_data else None
        
        message += "*🥘 Gizzard Stock Discrepancy:*\n"
        if gizzard_weight > 0 and gizzard_inventory_balance is not None:
            message += f"• Specification Sheet Gizzard: {gizzard_weight:,.2f} kg\n"
            message += f"• Inventory Records Gizzard: {gizzard_inventory_balance:,.2f} kg\n"
            message += f"• Discrepancy: {abs(gizzard_discrepancy):,.2f} kg {'more' if gizzard_discrepancy > 0 else 'less'} in specification sheet\n\n"
    
    # Get current time in WAT
    wat_tz = pytz.timezone('Africa/Lagos')
    current_time = datetime.now(pytz.UTC).astimezone(wat_tz)
    message += f"\n_Updated at: {current_time.strftime('%Y-%m-%d %I:%M:%S %p')} WAT_"
    
    return message

def send_combined_alert(webhook_url, stock_changes, stock_data, parts_changes, parts_data, inventory_data=None):
    """Send combined alert to Google Space."""
    try:
        # Only proceed if there are actual changes
        if not stock_changes and not parts_changes:
            print("No changes detected in either stock or parts. No alert needed.")
            return True
        
        message = "🔔 *Nasarawa Inventory Changes Detected*\n\n"
        print("Preparing combined changes message")
        
        # Add stock section if there are stock changes or if parts had changes
        if stock_changes or parts_changes:
            message += format_stock_section(stock_changes, stock_data, inventory_data)
            message += "\n"
        
        # Add parts section if there are parts changes or if stock had changes
        if parts_changes or stock_changes:
            message += format_parts_section(parts_changes, parts_data)
            message += "\n"
        
        # Add weight loss analysis section
        if inventory_data:
            message += format_weight_loss_section(inventory_data)
        
        # Get current time in WAT
        wat_tz = pytz.timezone('Africa/Lagos')
        current_time = datetime.now(pytz.UTC).astimezone(wat_tz)
        message += f"\n_Updated at: {current_time.strftime('%Y-%m-%d %I:%M:%S %p')} WAT_"
        
        payload = {
            "text": message
        }
        
        print("Sending webhook request...")
        response = requests.post(webhook_url, json=payload, timeout=10)  # Add timeout
        response.raise_for_status()  # Raise exception for bad status codes
        print(f"Webhook response status: {response.status_code}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error sending alert to Google Space: {str(e)}")
        return False

def send_discrepancy_alert(webhook_url, chicken_discrepancy, gizzard_discrepancy, stock_data, inventory_data):
    """Send discrepancy alert to Google Space."""
    try:
        message = format_discrepancy_alert(chicken_discrepancy, gizzard_discrepancy, stock_data, inventory_data)
        
        payload = {
            "text": message
        }
        
        print("Sending discrepancy alert webhook request...")
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        print(f"Discrepancy alert webhook response status: {response.status_code}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error sending discrepancy alert to Google Space: {str(e)}")
        return False

def main():
    try:
        # Get webhook URL from environment variable
        webhook_url = os.environ.get('SPACE_WEBHOOK_URL')
        if not webhook_url:
            raise ValueError("SPACE_WEBHOOK_URL environment variable not set")
        print("Webhook URL configured")

        # Initialize the Sheets API service
        print("Initializing Google Sheets service...")
        service = get_service()
        
        # Get current stock data
        stock_data = get_sheet_data(service, STOCK_SHEET_NAME, STOCK_RANGE)
        
        # Get current parts data
        parts_data = get_sheet_data(service, PARTS_SHEET_NAME, PARTS_RANGE)
        
        # Get complete inventory data for comparison and analysis
        inventory_data = get_inventory_data(service)
        
        # Load previous states
        previous_stock_data = load_previous_state(STOCK_STATE_FILE)
        previous_parts_data = load_previous_state(PARTS_STATE_FILE)
        
        # Load previous discrepancy states
        previous_chicken_discrepancy = load_previous_discrepancy_state(CHICKEN_DISCREPANCY_STATE_FILE)
        previous_gizzard_discrepancy = load_previous_discrepancy_state(GIZZARD_DISCREPANCY_STATE_FILE)
        
        # Calculate current discrepancies
        current_chicken_discrepancy = calculate_chicken_discrepancy(stock_data, inventory_data)
        current_gizzard_discrepancy = calculate_gizzard_discrepancy(stock_data, inventory_data)
        
        # Initialize flags for state updates
        stock_state_needs_update = True
        parts_state_needs_update = True
        
        # Check for changes in stock data
        stock_changes = []
        if not previous_stock_data:
            print("No previous stock state found, initializing stock state file...")
        else:
            print("Checking for stock changes...")
            stock_changes = detect_stock_changes(previous_stock_data, stock_data)
        
        # Check for changes in parts data
        parts_changes = []
        if not previous_parts_data:
            print("No previous parts state found, initializing parts state file...")
        else:
            print("Checking for parts changes...")
            parts_changes = detect_parts_changes(previous_parts_data, parts_data)
        
        # Check for discrepancy changes
        chicken_discrepancy_changed = detect_discrepancy_changes(
            previous_chicken_discrepancy, current_chicken_discrepancy, "chicken"
        )
        gizzard_discrepancy_changed = detect_discrepancy_changes(
            previous_gizzard_discrepancy, current_gizzard_discrepancy, "gizzard"
        )
        
        # Send combined alert if there are any changes
        if stock_changes or parts_changes:
            print("Changes detected, sending combined alert...")
            if send_combined_alert(webhook_url, stock_changes, stock_data, parts_changes, parts_data, inventory_data):
                print("Alert sent successfully, updating state files...")
            else:
                print("Failed to send alert, but will still update state files...")
        else:
            print("No changes detected in either stock or parts, updating state files...")
        
        # Send discrepancy alert if there are discrepancy changes
        if chicken_discrepancy_changed or gizzard_discrepancy_changed:
            print("Discrepancy changes detected, sending discrepancy alert...")
            if send_discrepancy_alert(webhook_url, current_chicken_discrepancy, current_gizzard_discrepancy, stock_data, inventory_data):
                print("Discrepancy alert sent successfully")
            else:
                print("Failed to send discrepancy alert")
        
        # Always update both state files at the end
        if stock_state_needs_update:
            save_current_state(stock_data, STOCK_STATE_FILE)
        if parts_state_needs_update:
            save_current_state(parts_data, PARTS_STATE_FILE)
        
        # Update discrepancy state files
        save_discrepancy_state(current_chicken_discrepancy, CHICKEN_DISCREPANCY_STATE_FILE)
        save_discrepancy_state(current_gizzard_discrepancy, GIZZARD_DISCREPANCY_STATE_FILE)

    except APIError as e:
        print(f"API Error: {str(e)}")
        # Don't exit with error to avoid GitHub Actions failure
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        # Don't exit with error to avoid GitHub Actions failure

if __name__ == '__main__':
    main() 