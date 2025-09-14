#!/usr/bin/env python3
"""
Threshold Update Demo
Demonstrates how changing the threshold file updates the Streamlit app automatically.
"""

import os
import time

def demo_threshold_updates():
    """Demo showing automatic threshold updates"""
    
    print("🎯 Threshold Update Demo")
    print("=" * 50)
    
    # Get threshold file path
    project_root = os.path.dirname(os.path.abspath(__file__))
    threshold_path = os.path.join(project_root, "models", "production", "optimal_threshold.txt")
    
    print(f"Threshold file: {threshold_path}")
    
    # Read current threshold
    with open(threshold_path, 'r') as f:
        original_threshold = float(f.read().strip())
    
    print(f"Original threshold: {original_threshold}")
    print("\n🌐 Make sure your Streamlit app is running at: http://localhost:8503")
    print("👀 Watch the sidebar 'Current Threshold' section for real-time updates!")
    
    # Demo different thresholds
    demo_thresholds = [0.5, 0.69770, 0.42, 0.75, original_threshold]
    
    for i, threshold in enumerate(demo_thresholds):
        print(f"\n📝 Step {i+1}: Setting threshold to {threshold}")
        
        # Update threshold file
        with open(threshold_path, 'w') as f:
            f.write(str(threshold))
        
        print(f"   ✅ Threshold updated to {threshold}")
        
        if threshold == 0.5:
            print("   📊 Effect: Balanced predictions")
        elif threshold == 0.69770:
            print("   📊 Effect: Conservative (minimize false alarms)")
        elif threshold == 0.42:
            print("   📊 Effect: Sensitive (catch more fraud)")
        elif threshold == 0.75:
            print("   📊 Effect: Very conservative")
        else:
            print("   📊 Effect: Back to original")
        
        print("   🔄 Check the Streamlit app - threshold should update automatically!")
        
        if i < len(demo_thresholds) - 1:
            print("   ⏳ Waiting 8 seconds before next update...")
            time.sleep(8)
    
    print(f"\n✅ Demo complete! Threshold restored to original value: {original_threshold}")
    print("\n🎉 Key Features Demonstrated:")
    print("   • Real-time threshold monitoring")
    print("   • Automatic file change detection")
    print("   • Visual update indicators")
    print("   • Built-in threshold editor")
    print("   • Auto-refresh options")

if __name__ == "__main__":
    try:
        demo_threshold_updates()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")