# Pipecat Phone Chatbot

This repository contains builds on top of the demo phone chatbot's simple dialin functionality by adding silence detection.

- **Simple dial-in**: Basic incoming call handling with silence detection built in.

## Getting Started

### Prerequisites

1. Create and activate a virtual environment:

   ```shell
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install requirements:

   ```shell
   pip install -r requirements.txt
   ```

3. Set up your environment variables:

   ```shell
   cp env.example .env
   ```

   Edit the `.env` file to include your API keys.

## Running the Example

### 1. Start the Bot Runner Service

The bot runner handles incoming requests and manages bot processes:

```shell
python bot_runner.py --host localhost
```

## Example: Simple Dial-in with Silence Detection

This example demonstrates basic handling of incoming calls without additional features like call transfer.

### Testing in Daily Prebuilt (No Actual Phone Calls)

```shell
curl -X POST "http://localhost:7860/start" \
	 -H "Content-Type: application/json" \
	 -d '{
		 "config": {
			"simple_dialin": {
			   "testInPrebuilt": true
			}
		 }
	  }'
```

This returns a Daily room URL where you can test the bot's basic conversation capabilities.

## Important Notes about the Silence Detection Implementation

The following changes have been made to the Simple_dialin.py implementation that is available on pipecat's github repo.
- Added a SilenceDetector class for detecting periods of silence more than 10s in duration
- When the first block of silence of duration > 10s is detected, the first silence prompt ("Hello, are you around?") is sent to the user.
- If no response from the user, the second silence prompt ("Are you still there? Call will end in 10 seconds if you don't respond.") is sent to the user.
- If the user still does not respond, the final silence prompt ("Terminating the call now. Goodbye!") is sent to the user and the call is set to be gravefully terminated.

I have also added a simple call summary log that contains information about the call, like the call ID, participant ID, number of silence prompts sent from the bot, and the overall call duration.