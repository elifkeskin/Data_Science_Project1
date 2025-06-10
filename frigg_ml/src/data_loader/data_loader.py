import os
from pathlib import Path
import logging
import pandas as pd
from typing import Optional, Union
from .error_messages import DataReadingErrorMessages as EM, SUPPORTED_FILE_EXTENSIONS

# Configure loading
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

data_reader_functions = {".csv": pd.read_csv, ".parquet": pd.read_parquet}


# The target is to create an object of this class for reusing.
class DataLoader:
    """A class for loading data from CSV files"""

    # Union provides a input check. In this case, file path can be a string or a Path object.
    def load_data(self, file_path: Union[str, Path]) -> Optional[pd.DataFrame]:
        """
        Loads data from the CSV file into a pandas DataFrame.

        Returns:
            Optional[pd.DataFrame]: A pandas DataFrame containing the loaded data,
                                     or None if an error occurs.

        Raises:
            TypeError: If the file path is not a string or a Path object.
            FileNotFoundError: If the file does not exist.
            ValueError: If the file path does not exist or if the file extension is not supported.

        """
        self.__validate_file_path(file_path)
        ext = self.__check_if_file_extension_is_supported(file_path)

        reader_func = data_reader_functions.get(ext)
        data: pd.DataFrame = reader_func(file_path)

        if data.empty:
            logger.error(EM.EMPTY_DATA_FILE.value)
            raise ValueError(EM.EMPTY_DATA_FILE.value)

        return data

    def __validate_file_path(self, file_path: Union[str, Path]) -> None:
        """
        Validates the file path.

        Args:
            file_path (Union[str, Path]): The file path to validate.

        Raises:
            TypeError: If the file path is not a string or Path object.
            FileNotFoundError: If the file does not exist or is empty.
        """

        # File path is not a string or Path object.In this case, file path is invalid.
        if not isinstance(file_path, (str, Path)):
            logger.error(
                EM.INVALID_FILE_PATH_TYPE.value.format(
                    type=type(file_path)
                )  # logging the message
            )
            raise TypeError(
                EM.INVALID_FILE_PATH_TYPE.value.format(
                    type=type(file_path)
                )  # raising an error
            )

        # File path is valid, but the file does not exist or is empty.
        if not os.path.exists(file_path):
            logger.error(EM.FILE_NOT_FOUND.value.format(file_path=file_path))
            raise FileNotFoundError(EM.FILE_NOT_FOUND.value.format(file_path=file_path))

    def __check_if_file_extension_is_supported(
        self, file_path: Union[str, Path]
    ) -> str:
        """
        Checks if the file extension is supported.

        Args:
            file_path (Union[str, Path]): The file path to check.

        Raises:
            ValueError: If the file extension is not supported.
        """
        ext = Path(file_path).suffix

        if ext not in SUPPORTED_FILE_EXTENSIONS:
            logger.error(
                EM.EXT_NOT_SUPPORTED.value.format(
                    ext=ext, supported_extensions=SUPPORTED_FILE_EXTENSIONS
                )
            )
            raise ValueError(
                EM.EXT_NOT_SUPPORTED.value.format(
                    ext=ext, supported_extensions=SUPPORTED_FILE_EXTENSIONS
                )
            )

        return ext
