import os
import time
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

# Configuration from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in .env file")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

def send_magic_link_emails_to_all_users():
    """
    Fetches all users from Supabase Auth and sends Magic Link emails using the predefined template.
    Handles pagination and rate limiting.
    """
    try:
        # Fetch users with pagination
        page = 1
        per_page = 100  # Adjust based on Supabase limits
        all_users = []

        while True:
            response = supabase.auth.admin.list_users(page=page, per_page=per_page)
            users = response  # Response is a list of user objects
            all_users.extend(users)
            if len(users) < per_page:
                break
            page += 1
            time.sleep(0.5)  # Avoid rate limits

        print(f"Found {len(all_users)} users. Sending Magic Link emails...")

        success_count = 0
        error_count = 0

        for user in all_users:
            email = user.email
            if not email:
                print(f"Skipping user {user.id}: No email found.")
                continue

            try:
                # Send Magic Link email using sign_in_with_otp
                response = supabase.auth.sign_in_with_otp({
                    "email": email,
                    "options": {
                        "email_redirect_to": "https://smart-notes-xu33.onrender.com/"  # Customize redirect URL
                    }
                })
                print(f"Magic Link email sent successfully to {email}")
                success_count += 1
                time.sleep(0.5)  # Rate limiting to avoid hitting Supabase limits
            except Exception as e:
                print(f"Error sending Magic Link to {email}: {str(e)}")
                error_count += 1

        print(f"\nSummary: {success_count} emails sent successfully, {error_count} failed.")

    except Exception as e:
        print(f"Error fetching users: {str(e)}")

if __name__ == "__main__":
    send_magic_link_emails_to_all_users()