from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import torch
from config import parse_config
from data_loader import DataBatchIterator
import numpy

def test_textcnn_model(model, test_data, config):
    model.eval()
    test_data_iter = iter(test_data)
    index=0
    true=[]
    pred=[]
    for idx, batch in enumerate(test_data_iter):
        model.zero_grad()
        ground_truth = batch.label
        outputs = model(batch.sent)
        result = torch.max(outputs, 1)[1]
        true.extend(ground_truth.data.numpy().tolist())
        pred.extend(result.data.numpy().tolist())
        index += 1
    size = index*32
    a = classification_report(true, pred)
    print(a)
    return


def main():
    config = parse_config()
    test_data = DataBatchIterator(
        config=config,
        is_train=False,
        dataset="test",
        # batch_size=config.batch_size)
    )
    test_data.load()
    model = torch.load('./results/model.pt')

    # 测试
    test_textcnn_model(model, test_data, config)



if __name__ == "__main__":
    main()
