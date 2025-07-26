import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from preprocessor import TextPreprocessor, load_ans_split_data
from model import SoftmaxRegression

TRAIN_PATH="data/new_train.tsv"
TEST_PATH="data/new_test.tsv"
VAL_SPLIT=0.2
MIN_WORD_FREQ=2

def calculate_accuracy(y_hat, y_true):
    _, predicted_labels=torch.max(y_hat, dim=1)
    correct_predictions=(predicted_labels==y_true).sum().item()
    total=len(y_true)
    return correct_predictions/total

def run_experiment(vectorization_method, learning_rate, epochs, batch_size):

    print("-"*60)
    print(f"Running experiment with vectorization method: {vectorization_method}, learning rate: {learning_rate}, epochs: {epochs}, batch size: {batch_size}")

    #加载和划分数据
    X_train, y_train, X_val, y_val, X_test, y_test=load_ans_split_data(TRAIN_PATH, TEST_PATH, val_split=VAL_SPLIT)

    preprocessor=TextPreprocessor(vectorization_method=vectorization_method, n_gram=2, min_freq=MIN_WORD_FREQ)
    preprocessor.fit(X_train, y_train)

    X_train_vec=preprocessor.transform(X_train)
    X_val_vec=preprocessor.transform(X_val)
    X_test_vec=preprocessor.transform(X_test)

    y_train_tensor=torch.LongTensor(y_train)
    y_val_tensor=torch.LongTensor(y_val)
    y_test_tensor=torch.LongTensor(y_test)

    input_dim=X_train_vec.shape[1]
    num_classes=preprocessor.num_classes
    model=SoftmaxRegression(input_dim, num_classes)

    #训练循环
    history={'train_loss':[], 'val_accuracy':[]}
    for epoch in range(epochs):
       model.train_loss_epoch=0
       permutaion=torch.randperm(X_train_vec.size(0))

       progress_bar=tqdm(range(0, X_train_vec.shape[0], batch_size), desc=f"Epoch {epoch+1}/{epochs}")

       for i in progress_bar:
          indices=permutaion[i:i+batch_size]
          X_batch, y_batch=X_train_vec[indices], y_train_tensor[indices]

          y_hat=model.forward(X_batch)
          loss=model.compute_loss(y_hat, y_batch)
          grad_W, grad_b=model.compute_gradient(X_batch, y_hat, y_batch)
          model.update_parameters(grad_W, grad_b, learning_rate)

          model.train_loss_epoch+=loss.item()

       avg_epoch_loss = model.train_loss_epoch/(len(X_train_vec) / batch_size)

       val_preds=model.forward(X_val_vec)
       val_acc=calculate_accuracy(val_preds, y_val_tensor)

       history['train_loss'].append(avg_epoch_loss)
       history['val_accuracy'].append(val_acc)

       print(f"Epoch {epoch+1} Summary:Train Loss={avg_epoch_loss:.4f}, Val_Accuracy={val_acc:.4f}")


    test_grads=model.forward(X_test_vec)
    test_acc=calculate_accuracy(test_grads, y_test_tensor)
    print(f"\nExperiment Finished. Final Test Accuracy:{test_acc:.4f}")
    print("-"*60)

    return history, test_acc, f"{vectorization_method.upper()}, LR={learning_rate}"

if  __name__ == "__main__":
    EPOCHS=20
    BATCH_SIZE=32

    experiment_configs=[
        {'method': 'bow', 'lr': 0.1},
        {'method': 'bow', 'lr': 0.5},
        {'method': 'ngram', 'lr': 0.1},
        {'method': 'ngram', 'lr': 0.5},
    ]
    results={}

    for config in experiment_configs:
        history, final_acc, label=run_experiment(
            vectorization_method=config['method'],
            learning_rate=config['lr'],
            epochs=EPOCHS,
            batch_size=BATCH_SIZE
        )
        results[label]={'history':history, 'final_acc':final_acc}

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2)=plt.subplots(1, 2, figsize=(16, 6))

    for label, res in results.items():
        ax2.plot(res['history']['val_accuracy'], label=label, marker='o', markersize=4)
    ax2.set_title("Validation Accuracy vs. Epochs", fontsize=16)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # 绘制最终测试准确率的条形图
    labels = list(results.keys())
    final_accuracies = [res['final_acc'] for res in results.values()]

    plt.figure(figsize=(10, 7))
    bars = plt.bar(labels, final_accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.title("Final Test Accuracy Comparison", fontsize=16)
    plt.xlabel("Experiment Configuration", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.ylim(0, max(final_accuracies) * 1.2)

    # 在条形图上显示数值
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom')

    plt.xticks(rotation=10)
    plt.show()
