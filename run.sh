
# Step 1: Prepare data
python clean_data.py

# Step 2: Train model
for i in `seq 1 10000` 
do
    python sa_search.py
done

# Step 3: Generate results
python compile_results.py
"C:/Program Files/Microsoft/R Open/R-3.5.3/bin/Rscript" --vanilla "visualization.R"
