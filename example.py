from pipeline import DamageAssessmentPipeline

# Initialize
pipeline = DamageAssessmentPipeline(
    model_path='best.pt',
    cost_table_path='part_costs.csv',
    hourly_rate=80.0
)

# Process an image
result = pipeline.process_image(
    image_path='test_car.jpg',
    vehicle_info={'make': 'Toyota', 'model': 'Camry', 'year': 2020}
)

# Print results
print(f"Found {result['num_damages']} damages")
print(f"Total cost: ${result['total_estimated_cost']:.2f}")

# Visualize
pipeline.visualize_results('./images/test1.jpg', result, output_path='result.jpg')