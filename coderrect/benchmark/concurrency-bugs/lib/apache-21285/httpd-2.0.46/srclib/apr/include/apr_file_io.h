/* ====================================================================
 * The Apache Software License, Version 1.1
 *
 * Copyright (c) 2000-2003 The Apache Software Foundation.  All rights
 * reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * 3. The end-user documentation included with the redistribution,
 *    if any, must include the following acknowledgment:
 *       "This product includes software developed by the
 *        Apache Software Foundation (http://www.apache.org/)."
 *    Alternately, this acknowledgment may appear in the software itself,
 *    if and wherever such third-party acknowledgments normally appear.
 *
 * 4. The names "Apache" and "Apache Software Foundation" must
 *    not be used to endorse or promote products derived from this
 *    software without prior written permission. For written
 *    permission, please contact apache@apache.org.
 *
 * 5. Products derived from this software may not be called "Apache",
 *    nor may "Apache" appear in their name, without prior written
 *    permission of the Apache Software Foundation.
 *
 * THIS SOFTWARE IS PROVIDED ``AS IS'' AND ANY EXPRESSED OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED.  IN NO EVENT SHALL THE APACHE SOFTWARE FOUNDATION OR
 * ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 * USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * ====================================================================
 *
 * This software consists of voluntary contributions made by many
 * individuals on behalf of the Apache Software Foundation.  For more
 * information on the Apache Software Foundation, please see
 * <http://www.apache.org/>.
 */

#ifndef APR_FILE_IO_H
#define APR_FILE_IO_H

/**
 * @file apr_file_io.h
 * @brief APR File I/O Handling
 */

#include "apr.h"
#include "apr_pools.h"
#include "apr_time.h"
#include "apr_errno.h"
#include "apr_file_info.h"
#include "apr_inherit.h"

#define APR_WANT_STDIO          /**< for SEEK_* */
#define APR_WANT_IOVEC          /**< for apr_file_writev */
#include "apr_want.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * @defgroup apr_file_io File I/O Handling Functions
 * @ingroup APR 
 * @{
 */

/**
 * @defgroup apr_file_open_flags File Open Flags/Routines
 * @{
 */

#define APR_READ       1           /**< Open the file for reading */
#define APR_WRITE      2           /**< Open the file for writing */
#define APR_CREATE     4           /**< Create the file if not there */
#define APR_APPEND     8           /**< Append to the end of the file */
#define APR_TRUNCATE   16          /**< Open the file and truncate to 0 length */
#define APR_BINARY     32          /**< Open the file in binary mode */
#define APR_EXCL       64          /**< Open should fail if APR_CREATE and file
                                        exists. */
#define APR_BUFFERED   128         /**< Open the file for buffered I/O */
#define APR_DELONCLOSE 256         /**< Delete the file after close */
#define APR_XTHREAD    512         /**< Platform dependent tag to open the file
                                        for use across multiple threads */
#define APR_SHARELOCK  1024        /**< Platform dependent support for higher
                                        level locked read/write access to support
                                        writes across process/machines */
#define APR_FILE_NOCLEANUP  2048   /**< Do not register a cleanup when the file
                                        is opened */
#define APR_SENDFILE_ENABLED  4096 /**< Advisory flag that this file should 
                                        support apr_sendfile operation */ 
/** @} */

/**
 * @defgroup apr_file_seek_flags File Seek Flags
 * @{
 */

/* flags for apr_file_seek */
/** Set the file position */
#define APR_SET SEEK_SET
/** Current */
#define APR_CUR SEEK_CUR
/** Go to end of file */
#define APR_END SEEK_END
/** @} */

/**
 * @defgroup apr_file_attrs_set_flags File Attribute Flags
 * @{
 */

/* flags for apr_file_attrs_set */
#define APR_FILE_ATTR_READONLY   0x01          /**< File is read-only */
#define APR_FILE_ATTR_EXECUTABLE 0x02          /**< File is executable */
/** @} */

/** File attributes */
typedef apr_uint32_t apr_fileattrs_t;

/** should be same as whence type in lseek, POSIX defines this as int */
typedef int       apr_seek_where_t;

/**
 * Structure for referencing files.
 */
typedef struct apr_file_t         apr_file_t;

/* File lock types/flags */
/**
 * @defgroup apr_file_lock_types File Lock Types
 * @{
 */

#define APR_FLOCK_SHARED        1       /**< Shared lock. More than one process
                                           or thread can hold a shared lock
                                           at any given time. Essentially,
                                           this is a "read lock", preventing
                                           writers from establishing an
                                           exclusive lock. */
#define APR_FLOCK_EXCLUSIVE     2       /**< Exclusive lock. Only one process
                                           may hold an exclusive lock at any
                                           given time. This is analogous to
                                           a "write lock". */

#define APR_FLOCK_TYPEMASK      0x000F  /**< mask to extract lock type */
#define APR_FLOCK_NONBLOCK      0x0010  /**< do not block while acquiring the
                                           file lock */
/** @} */

/**
 * Open the specified file.
 * @param new_file The opened file descriptor.
 * @param fname The full path to the file (using / on all systems)
 * @param flag Or'ed value of:
 * <PRE>
 *         APR_READ              open for reading
 *         APR_WRITE             open for writing
 *         APR_CREATE            create the file if not there
 *         APR_APPEND            file ptr is set to end prior to all writes
 *         APR_TRUNCATE          set length to zero if file exists
 *         APR_BINARY            not a text file (This flag is ignored on 
 *                               UNIX because it has no meaning)
 *         APR_BUFFERED          buffer the data.  Default is non-buffered
 *         APR_EXCL              return error if APR_CREATE and file exists
 *         APR_DELONCLOSE        delete the file after closing.
 *         APR_XTHREAD           Platform dependent tag to open the file
 *                               for use across multiple threads
 *         APR_SHARELOCK         Platform dependent support for higher
 *                               level locked read/write access to support
 *                               writes across process/machines
 *         APR_FILE_NOCLEANUP    Do not register a cleanup with the pool 
 *                               passed in on the <EM>cont</EM> argument (see below).
 *                               The apr_os_file_t handle in apr_file_t will not
 *                               be closed when the pool is destroyed.
 *         APR_SENDFILE_ENABLED  Open with appropriate platform semantics
 *                               for sendfile operations.  Advisory only,
 *                               apr_sendfile does not check this flag.
 * </PRE>
 * @param perm Access permissions for file.
 * @param cont The pool to use.
 * @remark If perm is APR_OS_DEFAULT and the file is being created, appropriate 
 *      default permissions will be used.  *arg1 must point to a valid file_t, 
 *      or NULL (in which case it will be allocated)
 */
APR_DECLARE(apr_status_t) apr_file_open(apr_file_t **new_file, const char *fname,
                                   apr_int32_t flag, apr_fileperms_t perm,
                                   apr_pool_t *cont);

/**
 * Close the specified file.
 * @param file The file descriptor to close.
 */
APR_DECLARE(apr_status_t) apr_file_close(apr_file_t *file);

/**
 * delete the specified file.
 * @param path The full path to the file (using / on all systems)
 * @param cont The pool to use.
 * @remark If the file is open, it won't be removed until all instances are closed.
 */
APR_DECLARE(apr_status_t) apr_file_remove(const char *path, apr_pool_t *cont);

/**
 * rename the specified file.
 * @param from_path The full path to the original file (using / on all systems)
 * @param to_path The full path to the new file (using / on all systems)
 * @param pool The pool to use.
 * @warning If a file exists at the new location, then it will be overwritten.  
 *      Moving files or directories across devices may not be possible.
 */
APR_DECLARE(apr_status_t) apr_file_rename(const char *from_path, 
                                          const char *to_path,
                                          apr_pool_t *pool);

/**
 * copy the specified file to another file.
 * @param from_path The full path to the original file (using / on all systems)
 * @param to_path The full path to the new file (using / on all systems)
 * @param perms Access permissions for the new file if it is created.
 *     In place of the usual or'd combination of file permissions, the
 *     value APR_FILE_SOURCE_PERMS may be given, in which case the source
 *     file's permissions are copied.
 * @param pool The pool to use.
 * @remark The new file does not need to exist, it will be created if required.
 * @warning If the new file already exists, its contents will be overwritten.
 */
APR_DECLARE(apr_status_t) apr_file_copy(const char *from_path, 
                                        const char *to_path,
                                        apr_fileperms_t perms,
                                        apr_pool_t *pool);

/**
 * append the specified file to another file.
 * @param from_path The full path to the source file (using / on all systems)
 * @param to_path The full path to the destination file (using / on all systems)
 * @param perms Access permissions for the destination file if it is created.
 *     In place of the usual or'd combination of file permissions, the
 *     value APR_FILE_SOURCE_PERMS may be given, in which case the source
 *     file's permissions are copied.
 * @param pool The pool to use.
 * @remark The new file does not need to exist, it will be created if required.
 */
APR_DECLARE(apr_status_t) apr_file_append(const char *from_path, 
                                          const char *to_path,
                                          apr_fileperms_t perms,
                                          apr_pool_t *pool);

/**
 * Are we at the end of the file
 * @param fptr The apr file we are testing.
 * @remark Returns APR_EOF if we are at the end of file, APR_SUCCESS otherwise.
 */
APR_DECLARE(apr_status_t) apr_file_eof(apr_file_t *fptr);

/**
 * open standard error as an apr file pointer.
 * @param thefile The apr file to use as stderr.
 * @param cont The pool to allocate the file out of.
 * 
 * @remark The only reason that the apr_file_open_std* functions exist
 * is that you may not always have a stderr/out/in on Windows.  This
 * is generally a problem with newer versions of Windows and services.
 * 
 * The other problem is that the C library functions generally work
 * differently on Windows and Unix.  So, by using apr_file_open_std*
 * functions, you can get a handle to an APR struct that works with
 * the APR functions which are supposed to work identically on all
 * platforms.
 */
APR_DECLARE(apr_status_t) apr_file_open_stderr(apr_file_t **thefile,
                                          apr_pool_t *cont);

/**
 * open standard output as an apr file pointer.
 * @param thefile The apr file to use as stdout.
 * @param cont The pool to allocate the file out of.
 * 
 * @remark The only reason that the apr_file_open_std* functions exist
 * is that you may not always have a stderr/out/in on Windows.  This
 * is generally a problem with newer versions of Windows and services.
 * 
 * The other problem is that the C library functions generally work
 * differently on Windows and Unix.  So, by using apr_file_open_std*
 * functions, you can get a handle to an APR struct that works with
 * the APR functions which are supposed to work identically on all
 * platforms.
 */
APR_DECLARE(apr_status_t) apr_file_open_stdout(apr_file_t **thefile,
                                          apr_pool_t *cont);

/**
 * open standard input as an apr file pointer.
 * @param thefile The apr file to use as stdin.
 * @param cont The pool to allocate the file out of.
 * 
 * @remark The only reason that the apr_file_open_std* functions exist
 * is that you may not always have a stderr/out/in on Windows.  This
 * is generally a problem with newer versions of Windows and services.
 * 
 * The other problem is that the C library functions generally work
 * differently on Windows and Unix.  So, by using apr_file_open_std*
 * functions, you can get a handle to an APR struct that works with
 * the APR functions which are supposed to work identically on all
 * platforms.
 */
APR_DECLARE(apr_status_t) apr_file_open_stdin(apr_file_t **thefile,
                                              apr_pool_t *cont);

/**
 * Read data from the specified file.
 * @param thefile The file descriptor to read from.
 * @param buf The buffer to store the data to.
 * @param nbytes On entry, the number of bytes to read; on exit, the number of bytes read.
 * @remark apr_file_read will read up to the specified number of bytes, but 
 *         never more.  If there isn't enough data to fill that number of 
 *         bytes, all of the available data is read.  The third argument is 
 *         modified to reflect the number of bytes read.  If a char was put 
 *         back into the stream via ungetc, it will be the first character 
 *         returned. 
 *
 *         It is not possible for both bytes to be read and an APR_EOF or other 
 *         error to be returned.
 *
 *         APR_EINTR is never returned.
 */
APR_DECLARE(apr_status_t) apr_file_read(apr_file_t *thefile, void *buf,
                                   apr_size_t *nbytes);

/**
 * Write data to the specified file.
 * @param thefile The file descriptor to write to.
 * @param buf The buffer which contains the data.
 * @param nbytes On entry, the number of bytes to write; on exit, the number 
 *               of bytes written.
 * @remark apr_file_write will write up to the specified number of bytes, but never 
 *      more.  If the OS cannot write that many bytes, it will write as many 
 *      as it can.  The third argument is modified to reflect the * number 
 *      of bytes written. 
 *
 *      It is possible for both bytes to be written and an error to be returned.
 *
 *      APR_EINTR is never returned.
 */
APR_DECLARE(apr_status_t) apr_file_write(apr_file_t *thefile, const void *buf,
                                    apr_size_t *nbytes);

/**
 * Write data from iovec array to the specified file.
 * @param thefile The file descriptor to write to.
 * @param vec The array from which to get the data to write to the file.
 * @param nvec The number of elements in the struct iovec array. This must 
 *             be smaller than APR_MAX_IOVEC_SIZE.  If it isn't, the function 
 *             will fail with APR_EINVAL.
 * @param nbytes The number of bytes written.
 * @remark It is possible for both bytes to be written and an error to be returned.
 *      APR_EINTR is never returned.
 *
 *      apr_file_writev is available even if the underlying operating system 
 *
 *      doesn't provide writev().
 */
APR_DECLARE(apr_status_t) apr_file_writev(apr_file_t *thefile,
                                     const struct iovec *vec,
                                     apr_size_t nvec, apr_size_t *nbytes);

/**
 * Read data from the specified file, ensuring that the buffer is filled
 * before returning.
 * @param thefile The file descriptor to read from.
 * @param buf The buffer to store the data to.
 * @param nbytes The number of bytes to read.
 * @param bytes_read If non-NULL, this will contain the number of bytes read.
 * @remark apr_file_read will read up to the specified number of bytes, but never 
 *      more.  If there isn't enough data to fill that number of bytes, 
 *      then the process/thread will block until it is available or EOF 
 *      is reached.  If a char was put back into the stream via ungetc, 
 *      it will be the first character returned. 
 *
 *      It is possible for both bytes to be read and an error to be 
 *      returned.  And if *bytes_read is less than nbytes, an
 *      accompanying error is _always_ returned.
 *
 *      APR_EINTR is never returned.
 */
APR_DECLARE(apr_status_t) apr_file_read_full(apr_file_t *thefile, void *buf,
                                        apr_size_t nbytes,
                                        apr_size_t *bytes_read);

/**
 * Write data to the specified file, ensuring that all of the data is
 * written before returning.
 * @param thefile The file descriptor to write to.
 * @param buf The buffer which contains the data.
 * @param nbytes The number of bytes to write.
 * @param bytes_written If non-NULL, this will contain the number of bytes written.
 * @remark apr_file_write will write up to the specified number of bytes, but never 
 *      more.  If the OS cannot write that many bytes, the process/thread 
 *      will block until they can be written. Exceptional error such as 
 *      "out of space" or "pipe closed" will terminate with an error.
 *
 *      It is possible for both bytes to be written and an error to be 
 *      returned.  And if *bytes_written is less than nbytes, an
 *      accompanying error is _always_ returned.
 *
 *      APR_EINTR is never returned.
 */
APR_DECLARE(apr_status_t) apr_file_write_full(apr_file_t *thefile, const void *buf,
                                         apr_size_t nbytes, 
                                         apr_size_t *bytes_written);

/**
 * put a character into the specified file.
 * @param ch The character to write.
 * @param thefile The file descriptor to write to
 */
APR_DECLARE(apr_status_t) apr_file_putc(char ch, apr_file_t *thefile);

/**
 * get a character from the specified file.
 * @param ch The character to write.
 * @param thefile The file descriptor to write to
 */
APR_DECLARE(apr_status_t) apr_file_getc(char *ch, apr_file_t *thefile);

/**
 * put a character back onto a specified stream.
 * @param ch The character to write.
 * @param thefile The file descriptor to write to
 */
APR_DECLARE(apr_status_t) apr_file_ungetc(char ch, apr_file_t *thefile);

/**
 * Get a string from a specified file.
 * @param str The buffer to store the string in. 
 * @param len The length of the string
 * @param thefile The file descriptor to read from
 * @remark The buffer will be '\0'-terminated if any characters are stored.
 */
APR_DECLARE(apr_status_t) apr_file_gets(char *str, int len, apr_file_t *thefile);

/**
 * Put the string into a specified file.
 * @param str The string to write. 
 * @param thefile The file descriptor to write to
 */
APR_DECLARE(apr_status_t) apr_file_puts(const char *str, apr_file_t *thefile);

/**
 * Flush the file's buffer.
 * @param thefile The file descriptor to flush
 */
APR_DECLARE(apr_status_t) apr_file_flush(apr_file_t *thefile);

/**
 * duplicate the specified file descriptor.
 * @param new_file The structure to duplicate into. 
 * @param old_file The file to duplicate.
 * @param p The pool to use for the new file.
 * @remark *new_file must point to a valid apr_file_t, or point to NULL
 */         
APR_DECLARE(apr_status_t) apr_file_dup(apr_file_t **new_file,
                                      apr_file_t *old_file,
                                      apr_pool_t *p);

/**
 * duplicate the specified file descriptor and close the original
 * @param new_file The old file that is to be closed and reused
 * @param old_file The file to duplicate
 * @param p        The pool to use for the new file
 *
 * @remark new_file MUST point at a valid apr_file_t. It cannot be NULL
 */
APR_DECLARE(apr_status_t) apr_file_dup2(apr_file_t *new_file,
                                        apr_file_t *old_file,
                                        apr_pool_t *p);

/**
 * move the specified file descriptor to a new pool
 * @param new_file Pointer in which to return the new apr_file_t
 * @param old_file The file to move
 * @param p        The pool to which the descriptor is to be moved
 * @remark Unlike apr_file_dup2(), this function doesn't do an
 *         OS dup() operation on the underlying descriptor; it just
 *         moves the descriptor's apr_file_t wrapper to a new pool.
 * @remark The new pool need not be an ancestor of old_file's pool.
 * @remark After calling this function, old_file may not be used
 */
APR_DECLARE(apr_status_t) apr_file_setaside(apr_file_t **new_file,
                                            apr_file_t *old_file,
                                            apr_pool_t *p);

/**
 * Move the read/write file offset to a specified byte within a file.
 * @param thefile The file descriptor
 * @param where How to move the pointer, one of:
 * <PRE>
 *            APR_SET  --  set the offset to offset
 *            APR_CUR  --  add the offset to the current position 
 *            APR_END  --  add the offset to the current file size 
 * </PRE>
 * @param offset The offset to move the pointer to.
 * @remark The third argument is modified to be the offset the pointer
          was actually moved to.
 */
APR_DECLARE(apr_status_t) apr_file_seek(apr_file_t *thefile, 
                                   apr_seek_where_t where,
                                   apr_off_t *offset);

/**
 * Create an anonymous pipe.
 * @param in The file descriptor to use as input to the pipe.
 * @param out The file descriptor to use as output from the pipe.
 * @param cont The pool to operate on.
 */
APR_DECLARE(apr_status_t) apr_file_pipe_create(apr_file_t **in, apr_file_t **out,
                                          apr_pool_t *cont);

/**
 * Create a named pipe.
 * @param filename The filename of the named pipe
 * @param perm The permissions for the newly created pipe.
 * @param cont The pool to operate on.
 */
APR_DECLARE(apr_status_t) apr_file_namedpipe_create(const char *filename, 
                                               apr_fileperms_t perm, 
                                               apr_pool_t *cont);

/**
 * Get the timeout value for a pipe or manipulate the blocking state.
 * @param thepipe The pipe we are getting a timeout for.
 * @param timeout The current timeout value in microseconds. 
 */
APR_DECLARE(apr_status_t) apr_file_pipe_timeout_get(apr_file_t *thepipe, 
                                               apr_interval_time_t *timeout);

/**
 * Set the timeout value for a pipe or manipulate the blocking state.
 * @param thepipe The pipe we are setting a timeout on.
 * @param timeout The timeout value in microseconds.  Values < 0 mean wait 
 *        forever, 0 means do not wait at all.
 */
APR_DECLARE(apr_status_t) apr_file_pipe_timeout_set(apr_file_t *thepipe, 
                                               apr_interval_time_t timeout);

/** file (un)locking functions. */

/**
 * Establish a lock on the specified, open file. The lock may be advisory
 * or mandatory, at the discretion of the platform. The lock applies to
 * the file as a whole, rather than a specific range. Locks are established
 * on a per-thread/process basis; a second lock by the same thread will not
 * block.
 * @param thefile The file to lock.
 * @param type The type of lock to establish on the file.
 */
APR_DECLARE(apr_status_t) apr_file_lock(apr_file_t *thefile, int type);

/**
 * Remove any outstanding locks on the file.
 * @param thefile The file to unlock.
 */
APR_DECLARE(apr_status_t) apr_file_unlock(apr_file_t *thefile);

/**accessor and general file_io functions. */

/**
 * return the file name of the current file.
 * @param new_path The path of the file.  
 * @param thefile The currently open file.
 */                     
APR_DECLARE(apr_status_t) apr_file_name_get(const char **new_path, 
                                           apr_file_t *thefile);

/**
 * Return the data associated with the current file.
 * @param data The user data associated with the file.  
 * @param key The key to use for retreiving data associated with this file.
 * @param file The currently open file.
 */                     
APR_DECLARE(apr_status_t) apr_file_data_get(void **data, const char *key, 
                                           apr_file_t *file);

/**
 * Set the data associated with the current file.
 * @param file The currently open file.
 * @param data The user data to associate with the file.  
 * @param key The key to use for assocaiteing data with the file.
 * @param cleanup The cleanup routine to use when the file is destroyed.
 */                     
APR_DECLARE(apr_status_t) apr_file_data_set(apr_file_t *file, void *data,
                                           const char *key,
                                           apr_status_t (*cleanup)(void *));

/**
 * Write a string to a file using a printf format.
 * @param fptr The file to write to.
 * @param format The format string
 * @param ... The values to substitute in the format string
 * @return The number of bytes written
 */ 
APR_DECLARE_NONSTD(int) apr_file_printf(apr_file_t *fptr, 
                                        const char *format, ...)
        __attribute__((format(printf,2,3)));

/**
 * set the specified file's permission bits.
 * @param fname The file (name) to apply the permissions to.
 * @param perms The permission bits to apply to the file.
 * @warning Some platforms may not be able to apply all of the available 
 *      permission bits; APR_INCOMPLETE will be returned if some permissions 
 *      are specified which could not be set.
 *
 *      Platforms which do not implement this feature will return APR_ENOTIMPL.
 */
APR_DECLARE(apr_status_t) apr_file_perms_set(const char *fname,
                                           apr_fileperms_t perms);

/**
 * Set attributes of the specified file.
 * @param fname The full path to the file (using / on all systems)
 * @param attributes Or'd combination of
 * <PRE>
 *            APR_FILE_ATTR_READONLY   - make the file readonly
 *            APR_FILE_ATTR_EXECUTABLE - make the file executable
 * </PRE>
 * @param attr_mask Mask of valid bits in attributes.
 * @param cont the pool to use.
 * @remark This function should be used in preference to explict manipulation
 *      of the file permissions, because the operations to provide these
 *      attributes are platform specific and may involve more than simply
 *      setting permission bits.
 * @warning Platforms which do not implement this feature will return
 *      APR_ENOTIMPL.
 */
APR_DECLARE(apr_status_t) apr_file_attrs_set(const char *fname,
                                             apr_fileattrs_t attributes,
                                             apr_fileattrs_t attr_mask,
                                             apr_pool_t *cont);

/**
 * Create a new directory on the file system.
 * @param path the path for the directory to be created.  (use / on all systems)
 * @param perm Permissions for the new direcoty.
 * @param cont the pool to use.
 */                        
APR_DECLARE(apr_status_t) apr_dir_make(const char *path, apr_fileperms_t perm, 
                        apr_pool_t *cont);

/** Creates a new directory on the file system, but behaves like
 * 'mkdir -p'. Creates intermediate directories as required. No error
 * will be reported if PATH already exists.
 * @param path the path for the directory to be created.  (use / on all systems)
 * @param perm Permissions for the new direcoty.
 * @param pool the pool to use.
 */
APR_DECLARE(apr_status_t) apr_dir_make_recursive(const char *path,
                                                 apr_fileperms_t perm,
                                                 apr_pool_t *pool);

/**
 * Remove directory from the file system.
 * @param path the path for the directory to be removed.  (use / on all systems)
 * @param cont the pool to use.
 */                        
APR_DECLARE(apr_status_t) apr_dir_remove(const char *path, apr_pool_t *cont);

/**
 * get the specified file's stats.
 * @param finfo Where to store the information about the file.
 * @param wanted The desired apr_finfo_t fields, as a bit flag of APR_FINFO_ values 
 * @param thefile The file to get information about.
 */ 
APR_DECLARE(apr_status_t) apr_file_info_get(apr_finfo_t *finfo, 
                                          apr_int32_t wanted,
                                          apr_file_t *thefile);


/**
 * Truncate the file's length to the specified offset
 * @param fp The file to truncate
 * @param offset The offset to truncate to.
 */
APR_DECLARE(apr_status_t) apr_file_trunc(apr_file_t *fp, apr_off_t offset);

/**
 * Retrieve the flags that were passed into apr_file_open()
 * when the file was opened.
 * @return apr_int32_t the flags
 */
APR_DECLARE(apr_int32_t) apr_file_flags_get(apr_file_t *f);

/**
 * Get the pool used by the file.
 */
APR_POOL_DECLARE_ACCESSOR(file);

/**
 * Set a file to be inherited by child processes.
 *
 */
APR_DECLARE_INHERIT_SET(file);

/** @deprecated @see apr_file_inherit_set */
APR_DECLARE(void) apr_file_set_inherit(apr_file_t *file);

/**
 * Unset a file from being inherited by child processes.
 */
APR_DECLARE_INHERIT_UNSET(file);

/** @deprecated @see apr_file_inherit_unset */
APR_DECLARE(void) apr_file_unset_inherit(apr_file_t *file);

/**
 * Open a temporary file
 * @param fp The apr file to use as a temporary file.
 * @param templ The template to use when creating a temp file.
 * @param flags The flags to open the file with. If this is zero,
 *              the file is opened with 
 *              APR_CREATE | APR_READ | APR_WRITE | APR_EXCL | APR_DELONCLOSE
 * @param p The pool to allocate the file out of.
 * @remark   
 * This function  generates  a unique temporary file name from template.  
 * The last six characters of template must be XXXXXX and these are replaced 
 * with a string that makes the filename unique. Since it will  be  modified,
 * template must not be a string constant, but should be declared as a character
 * array.  
 *
 */
APR_DECLARE(apr_status_t) apr_file_mktemp(apr_file_t **fp, char *templ,
                                          apr_int32_t flags, apr_pool_t *p);

/** @} */

#ifdef __cplusplus
}
#endif

#endif  /* ! APR_FILE_IO_H */
