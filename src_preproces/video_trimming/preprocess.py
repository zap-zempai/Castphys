import pandas as pd

def remove_full_path_from_list_imgs_csv(images_df, final_path):
    # Remove full
    images_df.frame = images_df.frame.str.rsplit("\\", n= 1, expand=True)[1]
    # Remove .png extension
    images_df.frame = images_df.frame.str.split(".", n=1, expand=True)[0]
    # Conver tu int
    images_df.frame = images_df.frame.apply(lambda y: int(y))
    # Remove timestamp and Saving new csv
    images_df.drop(["timestamp"], axis=1).to_csv(final_path, header=False, index=False)

    return images_df
