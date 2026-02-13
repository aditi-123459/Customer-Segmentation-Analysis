#  Customer Segmentation Using KMeans Clustering

##  Project Overview
This project performs **customer segmentation** using the **KMeans clustering algorithm**.  
The goal is to group customers based on **income, spending patterns, and family structure** to better understand customer profiles.

The workflow includes **data cleaning, feature engineering, feature scaling, clustering, and visualization**.  
These steps help identify distinct customer segments for **targeted marketing and business strategy**.

## Dataset Description
The dataset contains customer demographic and spending data.

### Key Columns Used / Derived:
- **Income** – Customer’s annual income  
- **Recency** – Days since last purchase  
- **MntTotal** – Total spending on products  
- **MntRegularProds** – Spending on regular products  
- **Kidhome** – Number of kids in the household  
- **Teenhome** – Number of teens in the household  
- **Total_Children** – Derived feature (`Kidhome + Teenhome`)  
- **Cluster** – Cluster label assigned by KMeans  

### Total Records:
2205

### Total Features:
39

##  Technologies Used
- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  

## 🔎 Project Workflow

1. **Importing Required Libraries**  
2. **Loading Dataset** (`pd.read_csv`)  
3. **Preview Dataset** (`df.head()`)  
4. **Check Dataset Dimensions** (`df.shape`)  
5. **View Column Names** (`df.columns`)  
6. **Get Dataset Information** (`df.info()`)  
7. **Check Missing Values** (`df.isnull().sum()`)  
8. **Handle Missing Values** (`df['Income'].fillna(df['Income'].median(), inplace=True)`)  
9. **Feature Engineering**  
   - Create `Total_Children` as `Kidhome + Teenhome`  
10. **Select Features for Modeling**  
    ```python
    X = df[['Income', 'Recency', 'MntTotal', 'MntRegularProds', 'Total_Children']]
    ```  
11. **Scale Features** using `StandardScaler`  
12. **Determine Optimal Number of Clusters (Elbow Method)**  
    - Calculate WCSS for K = 1 to 10  
13. **Visualize Elbow Method** (`plt.plot(range(1,11), wcss)`)  
14. **Apply KMeans Clustering**  
    ```python
    kmeans = KMeans(n_clusters=4)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    ```  
15. **Check Cluster Distribution** (`df['Cluster'].value_counts()`)  
16. **Analyze Cluster Characteristics** (`df.groupby('Cluster').mean()`)  
17. **Visualize Clusters**  
    ```python
    plt.scatter(df['Income'], df['MntTotal'], c=df['Cluster'], cmap='viridis')
    ```

##  Insights & Visualizations

###  Cluster Analysis
- WCSS vs. number of clusters plot (Elbow Method) identified **4 clusters as optimal**  
- Cluster sizes were checked to ensure no cluster is disproportionately small  
- Cluster feature averages reveal distinct customer profiles:  
  - High-income, high-spending customers  
  - Low-income, low-spending customers  
  - Families with children and moderate spending  
  - Other groups with specific spending patterns  

###  Cluster Visualization
- Scatter plot of **Income vs Total Spending** shows clear separation of clusters  
- Clusters represent meaningful customer segments for **targeted marketing**  

##  Key Insights
- Customer groups have distinct **income and spending patterns**  
- Family size (`Total_Children`) affects spending behavior  
- KMeans clustering effectively segments customers into actionable groups  
- Visualizations make it easier to interpret cluster characteristics  

##  Conclusion
This project demonstrates a **complete data preprocessing, clustering, and analysis workflow**.  
Customer segments were identified, visualized, and analyzed, preparing the dataset for **business decision-making or marketing strategies**.

### Future Improvements
- Explore additional features (e.g., product types, recency of purchase)  
- Handle outliers in income or spending  
- Build predictive models using cluster labels as features  

##  Author
Aditi
