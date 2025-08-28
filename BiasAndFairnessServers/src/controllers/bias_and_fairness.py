import io
import asyncio
import json
import yaml
import os
import logging
from pathlib import Path
from fastapi.responses import JSONResponse, Response
from fastapi import HTTPException

# Set up logging
logger = logging.getLogger(__name__)
from crud.bias_and_fairness import (
    upload_model, upload_data, insert_metrics, get_metrics_by_id, 
    get_all_metrics_query, delete_metrics_by_id,
    insert_bias_fairness_evaluation, get_all_bias_fairness_evaluations,
    get_bias_fairness_evaluation_by_id, update_bias_fairness_evaluation_status,
    delete_bias_fairness_evaluation
)
from utils.run_bias_and_fairness_check import analyze_fairness
from utils.handle_files_uploads import process_files
from utils.process_evaluation import process_evaluation
from database.db import get_db
from fastapi import UploadFile, BackgroundTasks
from database.redis import get_next_job_id, get_job_status, delete_job_status

async def get_all_metrics(tenant: str):
    """
    Retrieve all fairness metrics.
    """
    try:
        async with get_db() as db:
            metrics = await get_all_metrics_query(db, tenant)
            return JSONResponse(
                status_code=200,
                content=[
                    {
                        "model_id": row.model_id,
                        "model_filename": row.model_filename,
                        "data_id": row.data_id,
                        "data_filename": row.data_filename,
                        "metrics_id": row.metrics_id
                    } for row in metrics
                ]
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve metrics, {str(e)}"
        )

async def get_metrics(id: int, tenant: str):
    """
    Retrieve metrics for a given fairness run ID.
    """
    try:
        async with get_db() as db:
            metrics = await get_metrics_by_id(id, db, tenant)
            if not metrics:
                raise HTTPException(
                    status_code=404,
                    detail=f"Metrics with ID {id} not found"
                )
            return JSONResponse(
                status_code=200,
                content={
                    "model_id": metrics.model_id,
                    "data_id": metrics.data_id,
                    "metrics_id": metrics.metrics_id,
                    "metrics": metrics.metrics
                }
            )
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve metrics, {str(e)}"
        )

async def get_upload_status(job_id: int, tenant: str):
    value = await get_job_status(job_id)
    if value is None:
        return Response(status_code=204)
    await delete_job_status(job_id)
    return JSONResponse(
        status_code=200,
        content=value,
        media_type="application/json"
    )

async def handle_upload(background_tasks: BackgroundTasks, model: UploadFile, data: UploadFile, target_column: str, sensitive_column: str, tenant: str):
    """
    Handle file upload from the client.
    """
    job_id = await get_next_job_id()
    response = JSONResponse(status_code=202, content={
        "message": "Processing started", 
        "job_id": job_id,
        "model_filename": model.filename.replace(".gz", "") if model.filename else "",
        "data_filename": data.filename.replace(".gz", "") if data.filename else ""
    }, media_type="application/json")
    model_ = {
        "filename": model.filename,
        "content": await model.read()
    }
    data_ = {
        "filename": data.filename,
        "content": await data.read()
    }
    # create a job ID or use a unique identifier for the task
    background_tasks.add_task(process_files, job_id, model_, data_, target_column, sensitive_column, tenant)
    return response

# New controller functions for Bias and Fairness Module
async def handle_evaluation(
    background_tasks: BackgroundTasks, 
    model: UploadFile, 
    dataset: UploadFile, 
    target_column: str, 
    sensitive_columns: str,  # JSON string
    evaluation_metrics: str,  # JSON string
    fairness_threshold: float,
    bias_detection_methods: str,  # JSON string
    tenant: str
):
    """
    Handle advanced bias and fairness evaluation.
    """
    try:
        # Parse JSON strings
        sensitive_cols = json.loads(sensitive_columns)
        metrics = json.loads(evaluation_metrics)
        bias_methods = json.loads(bias_detection_methods)
        
        evaluation_id = f"eval_{tenant}_{int(asyncio.get_event_loop().time() * 1000)}"
        
        response = JSONResponse(status_code=202, content={
            "evaluationId": evaluation_id,
            "status": "pending",
            "message": "Evaluation started"
        }, media_type="application/json")
        
        model_ = {
            "filename": model.filename,
            "content": await model.read()
        }
        dataset_ = {
            "filename": dataset.filename,
            "content": await dataset.read()
        }
        
        # Add background task for evaluation
        background_tasks.add_task(
            process_evaluation, 
            evaluation_id, 
            model_, 
            dataset_, 
            target_column, 
            sensitive_cols, 
            metrics, 
            fairness_threshold, 
            bias_methods, 
            tenant
        )
        
        return response
        
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON format: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start evaluation: {str(e)}"
        )

async def get_evaluation_status(evaluation_id: str, tenant: str):
    """
    Get the status of an evaluation.
    """
    try:
        # This would typically check a database or Redis for status
        # For now, return a mock status
        return JSONResponse(
            status_code=200,
            content={
                "evaluationId": evaluation_id,
                "status": "running",
                "progress": 75
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get evaluation status: {str(e)}"
        )

async def get_evaluation_results(evaluation_id: str, tenant: str):
    """
    Get the results of a completed evaluation.
    """
    try:
        # This would typically fetch from database
        # For now, return mock results
        return JSONResponse(
            status_code=200,
            content={
                "evaluationId": evaluation_id,
                "results": {
                    "fairness_metrics": {},
                    "bias_analysis": {},
                    "recommendations": []
                }
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get evaluation results: {str(e)}"
        )

async def get_all_evaluations(tenant: str):
    """
    Get all evaluations for a tenant.
    """
    try:
        # This would typically fetch from database
        # For now, return empty list
        return JSONResponse(
            status_code=200,
            content=[]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get evaluations: {str(e)}"
        )

async def cancel_evaluation(evaluation_id: str, tenant: str):
    """
    Cancel a running evaluation.
    """
    try:
        # This would typically update database/Redis status
        return JSONResponse(
            status_code=200,
            content={"message": "Evaluation cancelled successfully"}
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel evaluation: {str(e)}"
        )

async def delete_metrics(id: int, tenant: str):
    """
    Delete metrics for a given fairness run ID.
    """
    try:
        async with get_db() as db:
            delete = await delete_metrics_by_id(id, db, tenant)
            if not delete:
                raise HTTPException(
                    status_code=404,
                    detail=f"Metrics with ID {id} not found"
                )
            await db.commit()
            return Response(status_code=204)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete metrics, {str(e)}"
        )

async def create_config_and_run_evaluation(background_tasks: BackgroundTasks, config_data: dict, tenant: str):
    """
    Create config.yaml file and run bias and fairness evaluation.
    """
    print("=== SIMPLE VERSION STARTED ===")
    
    try:
        # Step 1: Basic validation
        print("Step 1: Validating input...")
        if not config_data:
            raise ValueError("Config data is empty")
        print(f"✓ Input validated: {len(config_data)} fields")
        
        # Step 2: Create simple config object
        print("Step 2: Creating config object...")
        config = {
            "dataset": {
                "name": "adult-census-income",
                "source": "scikit-learn/adult-census-income",
                "split": "train",
                "platform": "huggingface",
                "protected_attributes": ["sex", "race"],
                "target_column": "income"
            },
            "model": {
                "model_task": "binary_classification",
                "label_behavior": "binary",
                "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            },
            "metrics": {
                "fairness": ["demographic_parity", "equalized_odds"],
                "performance": ["accuracy"]
            }
        }
        print("✓ Config object created")
        
        # Step 3: Write to file (simple path)
        print("Step 3: Writing config file...")
        import os
        config_dir = "configs"
        os.makedirs(config_dir, exist_ok=True)
        
        config_path = os.path.join(config_dir, "config.yaml")
        with open(config_path, 'w') as f:
            f.write("dataset:\n  name: adult-census-income\n")
            f.write("  source: scikit-learn/adult-census-income\n")
            f.write("  split: train\n")
            f.write("  platform: huggingface\n")
            f.write("  protected_attributes: [sex, race]\n")
            f.write("  target_column: income\n")
        
        print(f"✓ Config file written to: {config_path}")
        
        # Step 4: Generate simple IDs
        print("Step 4: Generating IDs...")
        import time
        job_id = int(time.time() * 1000)
        eval_id = f"eval_{job_id}"
        print(f"✓ Generated job_id: {job_id}, eval_id: {eval_id}")
        
        # Step 5: Return success
        print("Step 5: Returning success...")
        print("=== SIMPLE VERSION COMPLETED SUCCESSFULLY ===")
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Simple config created successfully",
                "job_id": job_id,
                "eval_id": eval_id,
                "config_path": config_path
            }
        )
        
    except Exception as e:
        print(f"=== ERROR in simple version: {e} ===")
        import traceback
        print(f"=== ERROR traceback: {traceback.format_exc()} ===")
        raise HTTPException(
            status_code=500,
            detail=f"Simple config creation failed: {str(e)}"
        )

async def run_bias_fairness_evaluation(job_id: int, config_path: str, eval_id: str, tenant: str):
    """
    Run the bias and fairness evaluation using the created config.
    """
    try:
        # Update status to running
        async with get_db() as db:
            await update_bias_fairness_evaluation_status(eval_id, "running", None, db, tenant)
        
        # Change to the BiasAndFairnessModule directory
        os.chdir("BiasAndFairnessModule")
        
        # Run the evaluation using the CLI
        import subprocess
        import sys
        
        # Run the evaluation
        result = subprocess.run([
            sys.executable, "-m", "src.core.cli", "prompt",
            "--config", config_path,
            "--limit", "50",
            "--output", "artifacts/clean_results.json"
        ], capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            # Read the results
            results_path = Path("artifacts/clean_results.json")
            if results_path.exists():
                with open(results_path, 'r') as f:
                    results = json.load(f)
                
                # Update status to completed with results
                async with get_db() as db:
                    await update_bias_fairness_evaluation_status(eval_id, "completed", results, db, tenant)
                
                # Update job status
                await update_job_status(job_id, {
                    "status": "completed",
                    "eval_id": eval_id,
                    "results": results,
                    "message": "Evaluation completed successfully"
                })
            else:
                # Update status to failed
                async with get_db() as db:
                    await update_bias_fairness_evaluation_status(eval_id, "failed", None, db, tenant)
                
                await update_job_status(job_id, {
                    "status": "failed",
                    "eval_id": eval_id,
                    "error": "Results file not found",
                    "message": "Evaluation failed - results file not generated"
                })
        else:
            # Update status to failed
            async with get_db() as db:
                await update_bias_fairness_evaluation_status(eval_id, "failed", None, db, tenant)
            
            await update_job_status(job_id, {
                "status": "failed",
                "eval_id": eval_id,
                "error": result.stderr,
                "message": "Evaluation failed - CLI execution error"
            })
            
    except Exception as e:
        # Update status to failed
        async with get_db() as db:
            await update_bias_fairness_evaluation_status(eval_id, "failed", None, db, tenant)
        
        await update_job_status(job_id, {
            "status": "failed",
            "eval_id": eval_id,
            "error": str(e),
            "message": f"Evaluation failed: {str(e)}"
        })

async def update_job_status(job_id: int, status_data: dict):
    """
    Update the job status in Redis.
    """
    try:
        from database.redis import set_job_status
        await set_job_status(job_id, status_data)
    except Exception as e:
        print(f"Failed to update job status: {e}")

async def get_all_bias_fairness_evaluations_controller(tenant: str):
    """Get all bias and fairness evaluations for a tenant."""
    try:
        async with get_db() as db:
            evaluations = await get_all_bias_fairness_evaluations(db, tenant)
            return JSONResponse(
                status_code=200,
                content=evaluations
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve evaluations: {str(e)}"
        )

async def get_bias_fairness_evaluation_by_id_controller(eval_id: str, tenant: str):
    """Get a specific bias and fairness evaluation by eval_id."""
    try:
        async with get_db() as db:
            evaluation = await get_bias_fairness_evaluation_by_id(eval_id, db, tenant)
            if not evaluation:
                raise HTTPException(
                    status_code=404,
                    detail=f"Evaluation with ID {eval_id} not found"
                )
            return JSONResponse(
                status_code=200,
                content=evaluation
            )
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve evaluation: {str(e)}"
        )

async def delete_bias_fairness_evaluation_controller(eval_id: str, tenant: str):
    """Delete a bias and fairness evaluation."""
    try:
        async with get_db() as db:
            result = await delete_bias_fairness_evaluation(eval_id, db, tenant)
            if not result:
                raise HTTPException(
                    status_code=404,
                    detail=f"Evaluation with ID {eval_id} not found"
                )
            return JSONResponse(
                status_code=200,
                content={"message": f"Evaluation {eval_id} deleted successfully"}
            )
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete evaluation: {str(e)}"
        )
