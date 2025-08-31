import asyncio
import os
from dotenv import load_dotenv
from twikit import Client

# Load environment variables from .env file
load_dotenv()

async def main():
    # Get credentials from environment variables
    email = os.getenv('TWITTER_EMAIL')
    username = os.getenv('TWITTER_USERNAME')
    password = os.getenv('TWITTER_PASSWORD')
    
    if not all([email, username, password]):
        print("Error: Missing Twitter credentials in .env file")
        print("Please add the following variables to your .env file:")
        print("TWITTER_EMAIL=your_email@example.com")
        print("TWITTER_USERNAME=your_username")
        print("TWITTER_PASSWORD=your_password")
        return
    
    client = Client('en-US')
    
    try:
        await client.login(
            auth_info_1=email,
            auth_info_2=username,
            password=password
        )
        
        # Save cookies so we don't need to login every time
        client.save_cookies('cookies.json')
        print("✅ Successfully authenticated and saved cookies!")
        
        # Test the login by fetching tweets from a user
        user = await client.get_user_by_screen_name("elonmusk")
        tweets = await client.get_user_tweets(user.id, "Tweets", count=5)
        
        print(f"\n📱 Test: Successfully fetched {len(tweets)} tweets from @elonmusk")
        for i, tweet in enumerate(tweets, 1):
            print(f"{i}. {tweet.text[:100]}...")
            
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print("Please check your credentials in the .env file")

if __name__ == "__main__":
    asyncio.run(main())