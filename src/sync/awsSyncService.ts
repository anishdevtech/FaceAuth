import { AuthEvent } from '../storage/authSync';

/**
 * AWS API Gateway Endpoint URL.
 * Replace this with the Invoke URL of your deployed API Gateway + Lambda integration.
 * Example: 'https://xxxxxxx.execute-api.ap-south-1.amazonaws.com/prod/sync'
 */
const AWS_API_GATEWAY_URL = 'https://REPLACE_ME_WITH_YOUR_AWS_API_GATEWAY_URL';

/**
 * Uploads a single offline authentication event to AWS via a REST API Gateway.
 * If the API Gateway URL has not been configured, it falls back to a mock simulation
 * to prevent the app from failing during development.
 * 
 * @param event The authentication event from the offline WAQ.
 * @returns true if the event was successfully ingested by AWS.
 */
export async function uploadToAWS(event: AuthEvent): Promise<boolean> {
  // Fallback to simulation if URL is not yet configured by the user
  if (AWS_API_GATEWAY_URL === 'https://REPLACE_ME_WITH_YOUR_AWS_API_GATEWAY_URL') {
    return new Promise((resolve) => {
      setTimeout(() => {
        console.log(`[AWSSync] (MOCK) Uploaded event ${event.id} to AWS.`);
        resolve(true);
      }, 200);
    });
  }

  try {
    const response = await fetch(AWS_API_GATEWAY_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        // 'x-api-key': 'YOUR_API_KEY_HERE' // Uncomment if you secure API Gateway with a Usage Plan
      },
      body: JSON.stringify({
        id: event.id,
        userId: event.userId,
        name: event.name,
        timestamp: event.timestamp,
        success: event.success,
        type: event.type
      }),
    });

    if (!response.ok) {
      console.warn(`[AWSSync] HTTP Error ${response.status}: ${await response.text()}`);
      return false;
    }

    console.log(`[AWSSync] ✓ Successfully synced ${event.id} to AWS.`);
    return true;

  } catch (error) {
    console.warn(`[AWSSync] Network failure while syncing ${event.id} to AWS:`, error);
    return false;
  }
}
