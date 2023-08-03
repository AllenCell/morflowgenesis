from prefect import task

@task
def upload_img(img):
    print("Uploading image...", img)
    return img