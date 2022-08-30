import hashlib
import tempfile
import urllib
import requests
from tqdm import tqdm
import sys
import argparse
import time
import os 
import cv2
import tflite_runtime.interpreter as tflite
from urllib.request import urlopen, Request
from pathlib import Path
import shutil


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ext_delegate', 
        action='store_true',
        help = "Use accelarator")
    parser.add_argument(
        '--weight', 
        type =str,
        default = 'weight/ssd_mobilenet_v1_1_default_1.tflite')
    parser.add_argument(
        '--label',
        type =str, 
        default = 'utils/coco.yaml')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.56,
        help='confidence threshold')
    parser.add_argument(
        '--num_threads',
        type=int,
        default=None,
        help='Number of Threads')
    parser.add_argument(
        '--image',
        type = str,
        default = 'data/cat_dog.png',
        help = 'input image'
    )
    parser.add_argument(
        '--video',
        type = str,
        default = 'data/test_movie_3_shorts.mp4',
        help = 'input image'
    )
    parser.add_argument(
        '--input_mean',
        default = 127.5, type = float,
        help = 'input_mean'
    )
    parser.add_argument(
        '--input_std',
        default=127.5, type = float,
        help = "input standard deviation"
    )
    opt = parser.parse_args()
    return opt

def timefunc(func):
    @wraps(func)
    def timeit_wrapper(*args,**kwargs):
        start_time = time.perf_counter()
        result = func(*args,**kwargs)
        total_time = time.perf_counter() - start_time
        print(f'Function {func.__name__} took {total_time:0.6f} seconds')
        return result
    return timeit_wrapper

def load_model(path, ext_delegate=None,num_threads=None,armadillo=False):
    opt = parse_opt()
    if opt.ext_delegate and not armadillo:
        if os.path.isfile('/usr/lib/libvx_delegate.so'):
            ext_delegate = [tflite.load_delegate('/usr/lib/libvx_delegate.so')]
            print(f'loaded {ext_delegate}')
    try :
        if not Path(file).exists():
            attempt_download(file)
        interpreter = tflite.Interpreter(model_path=file,experimental_delegates=ext_delegate,num_threads=num_threads)
        return interpreter
    except (ValueError, NameError) as e:
        sys.stderr.write(f" Unable to find  \n{e}")



def download_url_to_file(url,dst, hash_prefix=None, progress=True):
    file_size = None
    req = Request(url, headers={"api_ver":"deepsort"})
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta,'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])
    
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False,dir=dst_dir)
    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(total = file_size, disable=not progress,
                unit= 'B',unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))
        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError('invalid hash value (expected "{}", got "{}")'
                                .format(hash_prefix, digest))
        shutil.move(f.name,dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)


def safe_download(file,url,url2=None,min_bytes=1E0,error_msg=''):
    """ Attempts to download file from url or url2, checks and removes incomplete downloads < min_bytes
    """
    file = Path(file)
    assert_msg = f"Download file '{file}' does not exist or size is <min_bytes={min_bytes}"
    try: #url1
        download_url_to_file(url,str(file))
        assert file.exists() and file.stat().st_size>min_bytes, assert_msg
    except Exception as e:#url2
        print(e)
        print(type(e))
        file.unlink(missing_ok=True) # remove partial download
        os.system(f"curl -L '{url2 or url}' -o '{file}' --retry 3 -C -")
    finally:
        if not file.exists() or file.stat().st_size < min_bytes:
            file.unlink(missing_ok=True)
            sys.stderr.write(f" ERROR:\n{error_msg}\n")


def attempt_download(file, repo="nixondutt/DeepSort_for_Edge_Device", release = 'v1.0'):
    print(f"Downloading {file}...")
    def github_assets(repository, version='latest'):
        # Return GitHub repo tag and assets
        if version != 'latest':
            version = f'tags/{version}'  # i.e. tags/v1.0
        response = requests.get(f'https://api.github.com/repos/{repository}/releases/{version}').json()  # github api
        return response['tag_name'], [x['name'] for x in response['assets']]  # tag, assets 

    file = Path(str(file).strip().replace("'",''))
    if not file.exists():
        name = Path(urllib.parse.unquote(str(file))).name  # decode '%2F' to '/' etc.
        if str(file).startswith(('http:/','https:/')):
            url = str(file).replace(':/', '://')  # Pathlib turns :// -> :/
            file = name.split('?')[0] # parse authentication 
            if Path(file).is_file():
                print(f'Found {url} locally at {file}')
            else:
                safe_download(file=file,url=url,min_bytes=1E5)
            return file

        try:
            tag, assets = github_assets(repo, release)
        except Exception as e:
            try:
                tag, assets = github_assets(repo) #latest release
            except Exception:
                try:
                    tag =  subprocess.check_output('git tag', shell=True, stderr=subprocess.STDOUT).decode().split()[-1]
                except Exception:
                    tag = release

        file.parent.mkdir(parents=True, exist_ok=True)
        safe_download(
            file,
            url=f'https://github.com/{repo}/releases/download/{tag}/{name}',
            min_bytes=1E5,
            error_msg=f'{file} missing, try downloading from https://github.com/{repo}/releases/{tag} or contact support team')
        

    return str(file)