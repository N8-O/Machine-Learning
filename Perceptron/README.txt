Perceptron code

# run standard perceptron algorithm
T = 10
r = 0.5
s_w = StandardPerceptron(train_x,train_label,T,r)
std_train_predictions = StandardPrediction(train_x,s_w)

# run voted perceptron algorithm
T = 10
r = .5
v_w,c = VotedPerceptron(train_x,train_label,T,r)
vt_train_predictions = VotedPrediction(train_x,v_w,c)

# run average perceptron algorithm
T = 10
r = 0.5
a = AveragePerceptron(train_x,train_label,T,r)
avg_train_predictions = StandardPrediction(train_x,a)
