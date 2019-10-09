from src.pci_crackdown_functions import * 

compile_results(data_path = 'data/output/training_data_tam_19890427.pkl', model = 'Results/models/best', include_text = 0 , output = "Results/data/training_df.xlsx")

compile_results(data_path = 'data/output/testing_data_tam_19890427.pkl', model = 'Results/models/best', include_text = 0 , output = "Results/data/testing_df.xlsx")

compile_results(data_path = 'data/output/prediction_data_HK2014.pkl', model = 'Results/models/best', include_text = 0 , output = "Results/data/predict_df_HK2014.xlsx")

compile_results(data_path = 'data/output/prediction_data_HK2019.pkl', model = 'Results/models/best', include_text = 0 , output = "Results/data/predict_df_HK2019.xlsx")


compile_results(data_path = 'data/output/training_data_tam_19890427.pkl', model = 'Results/models/best', include_text = 1 , output = "Results/data/with text/training_df.xlsx")

compile_results(data_path = 'data/output/testing_data_tam_19890427.pkl', model = 'Results/models/best', include_text = 1 , output = "Results/data/with text/testing_df.xlsx")

compile_results(data_path = 'data/output/prediction_data_HK2014.pkl', model = 'Results/models/best', include_text = 1 , output = "Results/data/with text/predict_df_HK2014.xlsx")

compile_results(data_path = 'data/output/prediction_data_HK2019.pkl', model = 'Results/models/best', include_text = 1 , output = "Results/data/with text/predict_df_HK2019.xlsx")

