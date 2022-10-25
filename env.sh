#!/bin/bash
if [ -z ${CDI_DIR+x} ] ; then
	SOURCE="${BASH_SOURCE[0]}"
	while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
	  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
	  SOURCE="$(readlink "$SOURCE")"
	  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
	done
	export CDI_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
	export PATH+=:$CDI_DIR/bin:$CDI_DIR/script:$CDI_DIR/tensorflow
	if [ $(uname) = "Darwin" ]; then
		export DYLD_LIBRARY_PATH+=:$CDI_DIR/lib
	else
		export LD_LIBRARY_PATH+=:$CDI_DIR/lib
	fi
	alias cdicd='cd $CDI_DIR'
	alias cdimake='cd $CDI_DIR; make; cd -'
fi
