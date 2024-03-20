import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import datetime

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird",
    "boat", "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

ADE_CLASSES = [
    "void", "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed ", "windowpane",
    "grass", "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain", "plant",
    "curtain", "chair", "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror",
    "rug", "field", "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub",
    "railing", "cushion", "base", "box", "column", "signboard", "chest of drawers", "counter",
    "sand", "sink", "skyscraper", "fireplace", "refrigerator", "grandstand", "path", "stairs",
    "runway", "case", "pool table", "pillow", "screen door", "stairway", "river", "bridge",
    "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill", "bench", "countertop",
    "stove", "palm", "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine",
    "hovel", "bus", "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight",
    "booth", "television receiver", "airplane", "dirt track", "apparel", "pole", "land",
    "bannister", "escalator", "ottoman", "bottle", "buffet", "poster", "stage", "van", "ship",
    "fountain", "conveyer belt", "canopy", "washer", "plaything", "swimming pool", "stool",
    "barrel", "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball", "food",
    "step", "tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher",
    "screen", "blanket", "sculpture", "hood", "sconce", "vase", "traffic light", "tray", "ashcan",
    "fan", "pier", "crt screen", "plate", "monitor", "bulletin board", "shower", "radiator",
    "glass", "clock", "flag"
]
DG_CLASSES = [
    "unknown",
    "urban",
    "agriculture",
    "rangeland",
    "forest",
    "water",
    "barren"
]
IS_CLASSES = [

    'Background',
    'Baseball Diamond',
    'Basketball Court',
    'Bridge',
    'Ground Track Field',
    'Harbor',
    'Helicopter',
    'Large Vehicle',
    'Plane',
    'Roundabout',
    'Ship',
    'Small Vehicle',
    'Soccer Ball Field',
    'Storage Tank',
    'Swimming Pool',
    'Tennis Court'
]
VH_CLASSES = (
    'clutter',
    'building',
    'car',
    'impervious_surface',
    'low_vegetation',
    'tree')

PD_CLASSES = (
    'clutter',
    'building',
    'car',
    'impervious_surface',
    'low_vegetation',
    'tree')


class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()


class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """

    def __init__(self, n_classes, dataset):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

        if dataset == 'voc':
            self.CLASSES = VOC_CLASSES
        elif dataset == 'ade':
            self.CLASSES = ADE_CLASSES
        elif dataset == 'deepglobe':
            self.CLASSES = DG_CLASSES
        elif dataset == 'iSAID':
            self.CLASSES = IS_CLASSES
        elif dataset == 'vaihingen':
            self.CLASSES = VH_CLASSES
        elif dataset == 'potsdam':
            self.CLASSES = PD_CLASSES
        else:
            NotImplementedError

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())

    def to_str(self, results, out_matrix=False, opts=None):
        string = "\n"
        for k, v in results.items():
            if k != "Class IoU" and k != "Class Acc" and k != 'confusion matrix':
                string += "%s: %f\n" % (k, v)

        string += 'Class IoU/Acc:\n'
        for (k, v1), v2 in zip(results['Class IoU'].items(), results['Class Acc'].values()):
            string += "  %s: %s (miou) , %s (acc) \n" % (self.CLASSES[k], str(v1), str(v2))
        if out_matrix is True:
            print_matrix = results['confusion matrix']
            # string += 'confusion matrix:'
            # for class_acc in print_matrix:
            #     string += '\n'
            #     for acc in class_acc:
            #         string += '%10.4e%%  ' % acc.item()
            # string += '\n'
            # now = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            with open(f"results/{opts.model}_{opts.dataset}_{opts.task}_{opts.now}.csv", "a+") as f:
                matrix = ''
                for class_acc in print_matrix:
                    matrix += '\n'
                    # classes_iou = ','.join([str(val_score['Class IoU'].get(c, 'x')) for c in range(opts.num_classes)])
                    for acc in class_acc:
                        matrix += f'{str(acc.item())},'
                f.write(f"{opts.curr_step}\n{matrix}\n")

        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        EPS = 1e-6
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        gt_sum = hist.sum(axis=1)
        mask = (gt_sum != 0)
        acc_cls = np.diag(hist) / (hist.sum(axis=1) + EPS)
        cls_acc = dict(zip(range(self.n_classes), [acc_cls[i] if m else "X" for i, m in enumerate(mask)]))
        acc_cls = np.nanmean(acc_cls)

        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + EPS)
        # mean_iu = np.nanmean(iu[iu.nonzero()])
        mean_iu = np.nanmean(iu[mask])
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), [iu[i] if m else "X" for i, m in enumerate(mask)]))
        print_matrix = hist.astype('float')

        return {
            "Overall Acc": acc,
            "Mean Acc": acc_cls,
            "FreqW Acc": fwavacc,
            "Class Acc": cls_acc,
            "Mean IoU": mean_iu,
            "Class IoU": cls_iu,
            'confusion matrix': print_matrix
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    def confusion_matrix_to_fig(self):
        cm = self.confusion_matrix.astype('float') / (self.confusion_matrix.sum(axis=1)+0.000001)[:, np.newaxis]
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(title=f'Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')

        fig.tight_layout()
        return fig


class AverageMeter(object):
    """Computes average values"""

    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()

    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0] += val
            record[1] += 1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]
