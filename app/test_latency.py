import requests
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from statistics import mean, median

def create_test_image():
    """Create a test image (digit 5) for testing."""
    img = np.zeros((28, 28), dtype=np.uint8)
    img[5:20, 5:20] = 255
    image = Image.fromarray(img)
    image.save("test_digit.png")
    return "test_digit.png"

def test_prediction_latency(n_requests=100):
    """Test prediction latency over multiple requests."""
    image_path = create_test_image()
    latencies = []
    
    print(f"Making {n_requests} predictions...")
    for i in range(n_requests):
        start_time = time.time()
        
        files = {'file': open(image_path, 'rb')}
        response = requests.post("http://localhost:8000/predict/", files=files)
        
        if response.status_code == 200:
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            latencies.append(latency)
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1} requests...")
        else:
            print(f"Request {i + 1} failed with status code: {response.status_code}")
    
    return latencies

def plot_latencies(latencies):
    """Plot latency distribution."""
    plt.figure(figsize=(10, 6))
    plt.hist(latencies, bins=30, alpha=0.75)
    plt.axvline(np.median(latencies), color='r', linestyle='dashed', linewidth=2, label=f'Median: {np.median(latencies):.2f}ms')
    plt.axvline(np.mean(latencies), color='g', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(latencies):.2f}ms')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Frequency')
    plt.title('Prediction Latency Distribution')
    plt.legend()
    plt.savefig('latency_distribution.png')
    plt.close()

if __name__ == "__main__":
    print("Starting latency test...")
    latencies = test_prediction_latency()
    
    # Calculate statistics
    avg_latency = mean(latencies)
    med_latency = median(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    
    print("\nLatency Statistics (milliseconds):")
    print(f"Average: {avg_latency:.2f}")
    print(f"Median: {med_latency:.2f}")
    print(f"95th percentile: {p95_latency:.2f}")
    print(f"99th percentile: {p99_latency:.2f}")
    
    # Plot latency distribution
    plot_latencies(latencies)
    print("\nLatency distribution plot saved as 'latency_distribution.png'") 