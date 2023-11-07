# @copyright (c) 2002-2023 iProcure Limited. All rights reserved.

version_info = (0, 0, 1)

team_email = "technology@iprocu.re"

author_info = (('Natasha Bernard', 'natasha.bernard@iprocu.re'), )

package_license = "iProcure Limited"
package_info = "Data cleaning models"

__version__ = ".".join(map(str, version_info))
__author__ = ", ".join("{} <{}>".format(*info) for info in author_info)

__all__ = (
    '__author__',
    '__version__',
    'author_info',
    'package_license',
    'package_info',
    'team_email',
    'version_info',
)

import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))