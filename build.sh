#!/bin/bash
PACKAGE_VERSION=v0.2.3
PACKAGE=dcmtrans
PACKAGE_URL=git+file://$(pwd)@$PACKAGE_VERSION#egg=$PACKAGE
DIST_DIR=dist
BUILD_DIR=dist/build
# export PIP_EXISTS_ACTION="i" # to ignore existing package
LOCAL_PACKAGES="$PACKAGE"

GIT_HOST="http://$(cat git_secret)@10.0.4.52:3000"

if [ ! -d $DIST_DIR ]; then
    mkdir $DIST_DIR
fi

if [ ! -d $BUILD_DIR ]; then
    mkdir $BUILD_DIR
fi

# remove old version
for package in $LOCAL_PACKAGES; do
    OLD_PACKAGE=$(find $BUILD_DIR | grep $package)
    echo $OLD_PACKAGE
    if [[ -f $OLD_PACKAGE ]]; then
        echo "--> remove $OLD_PACKAGE"
        rm -f $OLD_PACKAGE
    fi
done

# build wheel of this package
(GIT_LFS_SKIP_SMUDGE=1 \
    pip wheel -w $BUILD_DIR $PACKAGE_URL
)

# remove unused packages
LAST=""
for f in $(ls $BUILD_DIR); do
    CURR=$(echo $f | cut -d "-" -f 1)
    if [[ $CURR = $LAST ]]; then
        echo "--> remove $f"
        rm -f $BUILD_DIR/$f
    fi
    LAST=$CURR
done

# zip wheels
cd $BUILD_DIR
zip -r ../$PACKAGE-$PACKAGE_VERSION.zip ./*
