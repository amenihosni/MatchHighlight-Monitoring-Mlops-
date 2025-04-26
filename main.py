import matplotlib.pyplot as plt
import os
import mlflow
from datetime import datetime
from prefect import flow
from prefect.logging import get_run_logger
from model_pipeline import (
    process_video,
    prepare_graph_data,
    train_models,
    evaluate_detections,
    visualize_results
)

@flow(name="soccer_analytics_pipeline")
def main(video_path: str = "163.mp4"):
    logger = get_run_logger()
    experiment_name = "Soccer_Analytics_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    mlflow.set_experiment(experiment_name)
    
    logger.info(f"\n=== Starting Soccer Analytics Pipeline ===")
    logger.info(f"MLflow Experiment: {experiment_name}\n")
    
    if not os.path.exists(video_path):
        logger.error(f"❌ Error: Video file not found at {video_path}")
        return
    
    # Process video
    with mlflow.start_run(run_name="Video_Processing"):
        logger.info("1. Processing video with YOLO...")
        try:
            csv_path = process_video(video_path, frame_skip=3)
            logger.info(f"✔ Detections saved to {csv_path}")
            mlflow.log_param("video_path", video_path)
            mlflow.log_artifact(csv_path)
        except Exception as e:
            logger.error(f"❌ Video processing failed: {str(e)}")
            raise
    
    # Prepare graph data
    with mlflow.start_run(run_name="Graph_Preparation", nested=True):
        logger.info("\n2. Preparing graph data...")
        try:
            graph_data, ball_speeds = prepare_graph_data(csv_path)
            logger.info(f"✔ Created {len(graph_data)} graph samples")
            logger.info(f"✔ Collected {len(ball_speeds)} ball speed measurements")
            mlflow.log_metric("num_graph_samples", len(graph_data))
            mlflow.log_metric("num_ball_speeds", len(ball_speeds))
        except Exception as e:
            logger.error(f"❌ Graph preparation failed: {str(e)}")
            raise
    
    # Evaluate detections
    with mlflow.start_run(run_name="Detection_Evaluation", nested=True):
        logger.info("\n3. Evaluating detections...")
        try:
            metrics = evaluate_detections(csv_path)
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {str(e)}")
            raise
    
    # Train models
    with mlflow.start_run(run_name="Model_Training", nested=True):
        logger.info("\n4. Training GNN models...")
        try:
            movement_model, refiner_model, losses = train_models(graph_data, epochs=20)
            logger.info("✔ Model training completed.")
            
            mlflow.pytorch.log_model(movement_model, "movement_model")
            mlflow.pytorch.log_model(refiner_model, "refiner_model")
            mlflow.log_metric("final_loss", losses[-1])
            
            plt.figure(figsize=(8, 5))
            plt.plot(losses)
            plt.title("Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            mlflow.log_figure(plt.gcf(), "training_loss.png")
            plt.close()
        except Exception as e:
            logger.error(f"❌ Model training failed: {str(e)}")
            raise
    
    # Visualize results
    with mlflow.start_run(run_name="Result_Visualization", nested=True):
        logger.info("\n5. Visualizing results...")
        try:
            visualize_results(graph_data, (movement_model, refiner_model), ball_speeds)
            logger.info("✔ Visualization completed and logged.")
        except Exception as e:
            logger.error(f"❌ Visualization failed: {str(e)}")
            raise

    logger.info("\n=== Pipeline completed successfully ===")

if __name__ == "__main__":
    main()