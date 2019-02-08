import json


def gen_caption():

    prename = "COCO_train2014_"
    input_json = "captions_train2014.json"

    output_dir = "train_captions/"
    caption_dict = dict()

    with open(input_json, "r") as fp:
        caption_dict = json.load(fp)

    for caption_vulk in caption_dict["annotations"]:
        image_id = caption_vulk["image_id"]
        id = caption_vulk["id"]
        caption = caption_vulk["caption"]
        temp_id = '{:0>12}'.format(str(image_id))

        with open(output_dir+prename+temp_id+".txt", "w") as fp:
            fp.write(caption)
            fp.close()

if __name__ == '__main__':
    gen_caption()
