# Pima Indians Diabetes Database

`diabetes.csv`: 768 patients, all female, at least 21 years old and of Pima Indian heritage. 268 (34.9 %) have diabetes.

| Column | Description |
|---|---|
| `Pregnancies` | Number of pregnancies |
| `Glucose` | Plasma glucose concentration, 2 h after an oral glucose tolerance test |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skin-fold thickness (mm) |
| `Insulin` | 2-hour serum insulin (µU/ml) |
| `BMI` | Body mass index (kg/m²) |
| `DiabetesPedigreeFunction` | Score for diabetes likelihood based on family history |
| `Age` | Age (years) |
| `Outcome` | Target: 1 = diabetes, 0 = no diabetes |

Several columns use `0` for missing values (e.g. `Insulin`, `SkinThickness`); the model uses them as-is.

## Source

Originally from the National Institute of Diabetes and Digestive and Kidney Diseases.

Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus.* In Proceedings of the Symposium on Computer Applications and Medical Care (pp. 261–265). IEEE Computer Society Press.
