/**
 * Module to hold Eilmer-specific exceptions.
 */

module lmrexceptions;

/**
 * Base Eilmer exception.
 *
 * This exception serves as a base class for all Eilmer exceptions.
 *
 * Author: RJG
 * Date: 2023-05-07
 */
class LmrException : Exception {
    @nogc
    this(string message, string file=__FILE__, size_t line=__LINE__,
         Throwable next=null)
    {
        super(message, file, line, next);
    }
}

/**
 * Class to signal N-K specific exceptions.
 *
 * Author: RJG
 * Date: 2023-05-07
 */
class NewtonKrylovException : LmrException {
    @nogc
    this(string message, string file=__FILE__, size_t line=__LINE__,
         Throwable next=null)
    {
        super(message, file, line, next);
    }
}
