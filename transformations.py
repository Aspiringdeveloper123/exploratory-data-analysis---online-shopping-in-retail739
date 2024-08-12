import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns
from scipy import stats
from scipy.stats import boxcox


class DataTransform:
   def __init__(self, df):
    self.df = df

   def _convert_to_category(self):
        # create a list `columns` with the names of the columns to convert
        columns = ['administrative', 'informational', 'product_related']
        # Use a for loop to iterate over this list
        for column in columns: 
           # Convert the 'administrative', 'informational', 'product_related' columns to categorical
           # apply the astype('category') to each column in the DataFrame
           self.df[column] = self.df[column].astype('category') 

# extracting information from the dataframe and its columns
# creating a class like this will be useful as these are common tasks for performing EDA on any dataset
class DataFrameInfo:
    def __init__(self, df):
        self.df = df

# Describe all columns in the DataFrame to check their data types
    def _describe_columns(self):
        return self.df.info()

# Extract statistical values: median, standard deviation and mean from the columns and the DataFrame
    def _extract_stats(self):
        numerical_columns = ['administrative_duration', 'informational_duration', 'product_related_duration', 'bounce_rates', 'exit_rates', 'page_values']
        computed_stats = {'Mean': {}, 'Std' : {}, 'Median': {}, 'Mode': {}}  #dictionary to store stats for the column and has keys for the mean, median and sd. Each key maps to a value
        for numerical in numerical_columns:
            # numerical_columns is a local variable to that function, so you don't need self in this case. 
            # if the variable was actually an attribute in the __init__ method, then you would need to use the self keyword to indicate it 
            computed_stats['Mean'][numerical] = self.df[numerical].mean() # accessess dictionary for the mean stats
            computed_stats['Std'][numerical] = self.df[numerical].std()
            computed_stats['Median'][numerical] = self.df[numerical].median()
            computed_stats['Mode'][numerical] = self.df[numerical].mode()
        return computed_stats

# Count distinct values in categorical columns to understand the variety of categories present in the column
    def _count_distinct_values(self):
        categorical_columns = ['administrative', 'informational', 'product_related', 'month', 'operating_systems', 'browser', 'region', 'traffic_type', 'visitor_type', 'weekend', 'revenue']
        node_labels = []
        num_categorical_vals_per_col = []
        for col in categorical_columns:
            uniques = self.df[col].unique().tolist()
            node_labels.extend(uniques)
            num_categorical_vals_per_col.append((col, len(uniques)))
        return node_labels, num_categorical_vals_per_col

# Print out the shape of the DataFrame
    def _shape_of_df(self):
        shape = self.df.shape
        return shape
   
# Generate a count/percentage count of NULL values in each column
    def _count_nulls(self):
        print('Percentage of null values in each column:')
        null_percentage = (self.df.isnull().sum()/len(self.df)) * 100
        return null_percentage
    
# Calculate the skewness of  numerical columns
    def _skewness_of_cols(self):
        numerical_columns = ['administrative_duration', 'informational_duration', 'product_related_duration', 'bounce_rates', 'exit_rates', 'page_values']
        skewness = self.df[numerical_columns].skew()
        return skewness

# Return columns that are skewed by adding a threshold over which a column will be considered skewed
    def _check_skewness(self):
        # call the method using self (in the same class)
        skewness_cols = self._skewness_of_cols()
        skewed_columns = {column : skew for column, skew in skewness_cols.items() if skew > 1 or skew < -1}
        return skewed_columns

# Task 3: Impute missing values in the data

class DataFrameTransform:
    def __init__(self, df):
        self.df = df
  
    def _impute_columns(self):
        # create a dictionary where keys are column names and values are the imputation strategies (mean or mode)
        impute_strategies = {
            'administrative': 'mode',
            'administrative_duration': 'mean',
            'informational_duration': 'mean',
            'product_related': 'mode',
            'product_related_duration': 'mean',
           'operating_systems': 'mode' 
        }
        # initiates a loop that iterates over each column-strategy pair in the impute_strategies dictionary
        # fill in the missing values with either the mean or mode
        for column, strategy in impute_strategies.items():
            if strategy == 'mean':
                self.df[column] = self.df[column].fillna(self.df[column].mean())  
            elif strategy == 'mode':
                self.df[column] = self.df[column].fillna(self.df[column].mode()[0])

        return impute_strategies
    
    #Log transformations are used to reduce skewness in data by compressing the range of values, particularly useful when dealing with positively skewed distributions.
    def _log_transform(self, skewed_columns):
        for column in skewed_columns:
            self.df[column] = self.df[column].map(lambda x: np.log(x + 1) if x > 0 else 0)

    #def _boxcox_transform(self, skewed_columns):
        #for column in skewed_columns:
            # Apply Box-Cox transformation only if all values are positive
            #if np.all(self.df[column] > 0):
                #self.df[column], _ = boxcox(self.df[column])

    def _apply_log_transformations(self, skewed_columns):
        self._log_transform(skewed_columns)
    #def _apply_box_cox_transformations(self, skewed_columns): 
        #self._boxcox_transform(skewed_columns)

    def _identify_outliers(self):
        # Identify numerical columns
        # np.number is shorthand for any kind of numerical data
        numerical_columns = self.df.select_dtypes(include=[np.number]).columns
        outlier_dict = {}

        for col in numerical_columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            print(f"For {col}:")
            print(f"Q1 (25th percentile): {Q1}")
            print(f"Q3 (75th percentile): {Q3}")
            print(f"IQR: {IQR}")

           # Identify outliers in numerical columns
            outliers = self.df[(self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))]
            # Store outliers in the dictionary
            outlier_dict[col] = outliers
            # Print outliers if they exist
            if not outliers.empty:
                print(f"Outliers for column '{col}':")
                print(outliers)
            else:
                print(f"No outliers for column '{col}'.")
        
        return outlier_dict

    def _remove_outliers(self):
        columns_to_filter = ['informational_duration', 'product_related_duration']
    
        for col in columns_to_filter:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
        # Print the initial shape before removing outliers
            print(f"Initial shape before removing outliers from '{col}': {self.df.shape}")
        # Define bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

        # Remove rows that have outliers in the specified columns, keeps only rows where it's true within bounds
            self.df = self.df[self.df[col].between(lower_bound, upper_bound)]

             # Print the shape after removing outliers (tuple)
            print(f"New shape after removing outliers from '{col}': {self.df.shape}")


class Plotter:
    def __init__(self, df, nulls_before, nulls_after, skewed_columns):
        self.df = df
        self.nulls_before = nulls_before   
        self.nulls_after = nulls_after
        self.skewed_columns = skewed_columns
    
    def _plot_removals_of_nulls(self):
        # Combine before and after null percentages into a DataFrame
        combined_nulls = pd.DataFrame({
            'Column names': self.nulls_before.index,  
            'Before Imputation': self.nulls_before.values,
            'After Imputation': self.nulls_after.values
        })
    
    # Create a bar plot
        fig = px.bar(
            combined_nulls,
            x='Column names',
            y=['Before Imputation', 'After Imputation'],
            title='Percentage of Null Values Before and After Imputation',
            labels={'value': 'Percentage of Null Values (%)', 'variable': 'Nulls being present before and after imputation'},
            height=400
        )
        fig.show()

    # Visualise the skewed data
    def _hist_skewed_cols(self):
        print("Skewed columns:", self.skewed_columns) 
        # Create a new figure for each column
        for column in self.skewed_columns:
            # plot histogram for each column using plt.figure() so it's plotted in a separate window
            plt.figure(figsize=(10, 5))

            # plots histogram for current column
            plt.hist(self.df[column], bins=50)
            plt.title(f'Histogram of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.show()
    
    # Visualise the transformed skewed data
    def _transformed_skewed_data(self):
        print("Transformed skewed columns:", self.skewed_columns)
        # Create a new figure for each column
        for column in self.skewed_columns:
           plt.figure(figsize=(10, 5))
           plt.hist(self.df[column], bins=50)
           plt.title(f'Histogram of Transformed {column}')
           plt.xlabel(column)
           plt.ylabel('Frequency')
           plt.show()

    def _graph_with_removed_outliers(self):
        columns_to_filter = ['informational_duration', 'product_related_duration']
        for column in columns_to_filter:
            plt.figure(figsize=(10, 5))
            sns.boxplot(y=self.df[column], color='green', showfliers=True)
            sns.swarmplot(y=self.df[column], color='black', size=5)
            plt.title(f'Boxplot After Outlier Removal of: {column}')
            plt.ylabel(column)
            plt.show()


if __name__ == "__main__":
    # Load the DataFrame 
    loaded_df = pd.read_csv('customer_activity_data.csv')
    print("Data loaded from CSV successfully")
   
   # Initialize DataTransform and apply transformations
    transformer = DataTransform(loaded_df)
    transformer._convert_to_category()
    print("Columns converted to categorical")

   # Initialize DataFrameInfo and apply transformations - describe columns, extract stats, count distinct values,  shape of df, % of nulls
    info = DataFrameInfo(loaded_df)

    print("Describing columns:")
    describe_columns = info._describe_columns()
    print(describe_columns)

    print("Display statistics:")
    stats = info._extract_stats()
    print(stats)

    print("Count distinct values in categorical columns:")
    distinct_values = info._count_distinct_values()
    print(distinct_values)

    print("Shape of the dataframe:")
    shape_of_data = info._shape_of_df()
    print(shape_of_data)

    print("Percentage of nulls:")
    nulls_before_imputation = info._count_nulls()
    print(nulls_before_imputation)

    #Initialise DataFrameTransform and impute columns
    df_transform = DataFrameTransform(loaded_df)
    df_transform._impute_columns()
    print("Missing values imputed")

    # Run your NULL checking method/function again to check that all NULLs have been removed
    print("Percentage of nulls after imputation:")
    nulls_after_imputation = info._count_nulls()
    print(nulls_after_imputation)

    # Save the DataFrame to a CSV file
    loaded_df.to_csv('dtype_and_nulls_handled_customer_activity.csv', index=False)
    print("DataFrame saved to CSV successfully")

     # Load the cleaned DataFrame for transformation tasks
    cleaned_df = pd.read_csv('dtype_and_nulls_handled_customer_activity.csv')
    print("Cleaned data loaded from CSV successfully")

     # Task 4: Identify the skewed columns in the data
     # Initialize DataFrameInfo and check skewness
    info_cleaned = DataFrameInfo(cleaned_df)    # creating another instance here
    skewness_result = info_cleaned._skewness_of_cols()
    print("Skewness of numerical columns:")
    print(skewness_result)

    # Check which columns are skewed
    skewed_columns = info_cleaned._check_skewness()
    print("Columns considered skewed:")
    print(skewed_columns)

    # Apply the log transformations to reduce skewness
    log_transform = df_transform._apply_log_transformations(skewed_columns)   
    print("Log transformation applied to reduce skewness")

    # Apply the boxcox transformations to reduce skewness
    # boxcox_transform = df_transform._apply_box_cox_transformations(skewed_columns)
    # print("Boxcox transformation applied to reduce skewness")

    # Check the skewness after transformations
    skewed_columns_list = list(skewed_columns.keys())
    # Calculate skewness of the transformed DataFrame
    transformed_skewness = df_transform.df[skewed_columns_list].skew()
    print("Transformed skewness:")
    print(transformed_skewness)

    # Initialise Plotter and apply transformations
    plotter = Plotter(loaded_df, nulls_before_imputation, nulls_after_imputation, skewed_columns)
    plotter._plot_removals_of_nulls()
    print("Bar chart displayed to visualise Nulls before and after imputation:")

    # visualise the skewed data in a histogram
    plotter._hist_skewed_cols()
    print("Histogram displayed to visualise skewness of data for numerical columns:")

    # Visualize the transformed skewed data 
    plotter._transformed_skewed_data()
    print("Histogram displayed to visualize transformed skewness of data for numerical columns:")

    # Save a copy of the DataFrame to a CSV file after the skew transformations
    # we have to use df_transform.df because it represents the DataFrame after skew handling transformations have been applied. 
    df_transform.df.to_csv('dtype_nulls_handled_and_skewness_transformation.csv', index=False)
    print("Transformed DataFrame saved to CSV successfully")

    # Task 5 - seeing the transformed skewed data will allow us to see if there are any outliers (refer to transformed_skewed_data method)

    # Identify outliers
    df_transform._identify_outliers()
    print("Outliers identified from the DataFrame")

    # Remove outliers and visualize the results
    df_transform._remove_outliers()
    plotter._graph_with_removed_outliers()
    print("Histograms displayed to visualize data after outlier removal:")
 