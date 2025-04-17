# app.py
from flask import Flask, render_template, request

app = Flask(__name__)

# Training Data
Rainfall = [800, 950, 700, 1100, 900, 780, 1000, 850, 1150, 720]
Fertilizer = [120, 135, 110, 150, 130, 115, 140, 125, 160, 105]
Pesticides = [10, 15, 8, 20, 12, 9, 18, 11, 22, 7]
y = [2.8, 3.5, 2.4, 4.2, 3.1, 2.6, 3.8, 3.0, 4.5, 2.2]

# Step 1: Build X matrix with intercept
n = len(y)
X = [[1, Rainfall[i], Fertilizer[i], Pesticides[i]] for i in range(n)]

# Step 2: Transpose X
X_T = [[X[j][i] for j in range(len(X))] for i in range(len(X[0]))]

# Step 3: X_T * X
X_T_X = [[sum(X_T[i][k] * X[k][j] for k in range(n)) for j in range(len(X[0]))] for i in range(len(X_T))]

# Step 4: Determinant and Inverse
def determinant(matrix):
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    return sum((-1)**c * matrix[0][c] * determinant([row[:c] + row[c+1:] for row in matrix[1:]]) for c in range(len(matrix)))

def inverse(matrix):
    det = determinant(matrix)
    size = len(matrix)
    cofactors = [[((-1)**(r+c)) * determinant([row[:c] + row[c+1:] for row in (matrix[:r] + matrix[r+1:])]) for c in range(size)]
                 for r in range(size)]
    adjugate = [[cofactors[j][i] for j in range(size)] for i in range(size)]
    return [[adjugate[r][c] / det for c in range(size)] for r in range(size)]

X_T_X_inv = inverse(X_T_X)

# Step 5: X_T * y
X_T_y = [[sum(X_T[i][k] * y[k] for k in range(n))] for i in range(len(X_T))]

# Step 6: B = (X_T_X)^-1 * (X_T * y)
B = [[sum(X_T_X_inv[i][k] * X_T_y[k][0] for k in range(len(X_T)))] for i in range(len(X_T_X_inv))]

@app.route("/")
def form():
    return render_template("forms.html")

@app.route("/predict", methods=["POST"])
def predict():
    rainfall = float(request.form["rainfall"])
    fertilizer = float(request.form["fertilizer"])
    pesticides = float(request.form["pesticides"])

    prediction = B[0][0] + B[1][0] * rainfall + B[2][0] * fertilizer + B[3][0] * pesticides
    prediction = round(prediction, 2)

    return render_template("results.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host='0.0.0,',port=5000)
