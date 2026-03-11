import os
from supabase import create_client, Client
import json

# Credentials from app.py
SUPABASE_URL = "https://tqypautmlmjjasvqwdgw.supabase.co"
SUPABASE_KEY = "sb_publishable_ALUruq3-ztjt48v46ZqiQw_-sZw4SHw"

def test_connection():
    print(f"Testing connection to: {SUPABASE_URL}")
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Try to simple select. Even if RLS blocks it or table is empty, 
        # getting a response means connection is good.
        # We try to select from 'profiles' since we asked user to create it.
        print("Attempting to fetch from 'profiles' table...")
        response = supabase.table('profiles').select("*").limit(1).execute()
        
        print("Connection Successful!")
        print(f"Response data: {response.data}")
        return True
    except Exception as e:
        print("\nConnection Failed!")
        print(f"Error: {str(e)}")
        # Check if it's a specific 'table not found' error which actually means connection worked
        if "relation \"public.profiles\" does not exist" in str(e):
             print("\nNOTE: The connection worked, but the 'profiles' table doesn't exist yet.")
             print("Please run the SQL schema provided earlier in the Supabase SQL Editor.")
             return True
        return False

if __name__ == "__main__":
    test_connection()
