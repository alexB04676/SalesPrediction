# Build the Docker image
build:
	docker build -t sales_prediction1.0 .

# Run the container
run:
	docker run --rm -it sales_prediction1.0

# Rebuild from scratch (no cache)
rebuild:
	docker build --no-cache -t sales-prediction .

# Clean up old images (if needed)
clean:
	docker rmi sales-prediction || true