import os
import mlflow
import matplotlib.pyplot as plt
from datetime import datetime
from model_pipeline import (
    process_video,
    prepare_graph_data,
    train_models,
    evaluate_detections,
    visualize_results
)
import mlflow
import os
from datetime import datetime

def main():
    experiment_name = "Soccer_Analytics_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    mlflow.set_experiment(experiment_name)
    
    print(f"\n=== Starting Soccer Analytics Pipeline ===")
    print(f"MLflow Experiment: {experiment_name}\n")
    
    video_path = r"163.mp4"
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found at {video_path}")
        return
    
    with mlflow.start_run(run_name="Video_Processing"):
        print("1. Processing video with YOLO...")
        try:
            csv_path = process_video(video_path, frame_skip=3)
            print(f"✔ Detections saved to {csv_path}")
            mlflow.log_param("video_path", video_path)
            mlflow.log_artifact(csv_path)
        except Exception as e:
            print(f"❌ Video processing failed: {str(e)}")
            return
    
    with mlflow.start_run(run_name="Graph_Preparation", nested=True):
        print("\n2. Preparing graph data...")
        try:
            graph_data, ball_speeds = prepare_graph_data(csv_path)
            print(f"✔ Created {len(graph_data)} graph samples")
            print(f"✔ Collected {len(ball_speeds)} ball speed measurements")
            mlflow.log_metric("num_graph_samples", len(graph_data))
            mlflow.log_metric("num_ball_speeds", len(ball_speeds))
        except Exception as e:
            print(f"❌ Graph preparation failed: {str(e)}")
            return
    
    with mlflow.start_run(run_name="Detection_Evaluation", nested=True):
        print("\n3. Evaluating detections...")
        try:
            metrics = evaluate_detections(csv_path)
        except Exception as e:
            print(f"❌ Evaluation failed: {str(e)}")
            return
    
    with mlflow.start_run(run_name="Model_Training", nested=True):
        print("\n4. Training GNN models...")
        try:
            movement_model, refiner_model, losses = train_models(graph_data, epochs=20)
            print("✔ Model training completed.")
            
            # Log models
            mlflow.pytorch.log_model(movement_model, "movement_model")
            mlflow.pytorch.log_model(refiner_model, "refiner_model")
            
            # Log final loss
            mlflow.log_metric("final_loss", losses[-1])
            
            # Plot training loss
            plt.figure(figsize=(8, 5))
            plt.plot(losses)
            plt.title("Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            mlflow.log_figure(plt.gcf(), "training_loss.png")
            plt.close()
        except Exception as e:
            print(f"❌ Model training failed: {str(e)}")
            return
    
    with mlflow.start_run(run_name="Result_Visualization", nested=True):
        print("\n5. Visualizing results...")
        try:
            visualize_results(graph_data, (movement_model, refiner_model), ball_speeds)
            print("✔ Visualization completed and logged.")
        except Exception as e:
            print(f"❌ Visualization failed: {str(e)}")
            return

    print("\n=== Pipeline completed successfully ===")

if __name__ == "__main__":
    main()