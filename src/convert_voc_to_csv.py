import os
import xml.etree.ElementTree as ET
import pandas as pd

def xml_to_csv(annotations_path, output_csv):
    xml_list = []
    for filename in os.listdir(annotations_path):
        if filename.endswith('.xml'):
            xml_path = os.path.join(annotations_path, filename)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for member in root.findall('object'):
                value = (
                    root.find('filename').text,
                    int(root.find('size/width').text),
                    int(root.find('size/height').text),
                    member.find('name').text,
                    int(member.find('bndbox/xmin').text),
                    int(member.find('bndbox/ymin').text),
                    int(member.find('bndbox/xmax').text),
                    int(member.find('bndbox/ymax').text)
                )
                xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    xml_df.to_csv(output_csv, index=False)
    print(f"Annotations saved to {output_csv}")

if __name__ == "__main__":
    annotations_path = 'data/annotations'
    output_csv = 'data/annotations.csv'
    xml_to_csv(annotations_path, output_csv)