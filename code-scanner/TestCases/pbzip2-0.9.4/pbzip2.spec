Name:		pbzip2
Version:	0.9.4
Release:	1%{?dist}
Summary:	Parallel implementation of bzip2
URL:		http://www.compression.ca/pbzip2/
License:	BSD
Group:		Applications/File
BuildRoot:	%{_tmppath}/%{name}-%{version}-%{release}-root-%(%{__id_u} -n)
BuildRequires:	bzip2-devel
Source0:	http://www.compression.ca/pbzip2/%{name}-%{version}.tar.gz

%description
PBZIP2 is a parallel implementation of the bzip2 block-sorting file
compressor that uses pthreads and achieves near-linear speedup on SMP
machines.  The output of this version is fully compatible with bzip2
v1.0.2 (ie: anything compressed with pbzip2 can be decompressed with
bzip2).


%prep
%setup -q


%build
make


%install
rm -rf %{buildroot}
install -D -m755 %{name} %{buildroot}%{_bindir}/%{name}
install -D -m644 %{name}.1 %{buildroot}%{_mandir}/man1/%{name}.1


%clean
rm -rf %{buildroot}


%files
%defattr(-,root,root)
%doc AUTHORS ChangeLog COPYING README
%{_bindir}/%{name}
%{_mandir}/man1/*


%changelog
* Thu Aug 30 2005 Jeff Gilchrist <pbzip2@compression.ca> - 0.9.4-1
- Updated RPM spec with suggestions from Oliver Falk

* Fri Jul 29 2005 Bryan Stillwell <bryan@bokeoa.com> - 0.9.3-1
- Release 0.9.3
- Removed non-packaging changelog info
- Added dist macro to release field
- Clean buildroot at the beginning of the install section
- Modified buildroot tag to match with Fedora PackagingGuidelines
- Shortened Requires and BuildRequires list
- Changed description to match with the Debian package

* Sat Mar 12 2005 Jeff Gilchrist <pbzip2@compression.ca> - 0.9.2-1
- Release 0.9.2

* Sat Jan 29 2005 Jeff Gilchrist <pbzip2@compression.ca> - 0.9.1-1
- Release 0.9.1

* Sun Jan 24 2005 Jeff Gilchrist <pbzip2@compression.ca> - 0.9-1
- Release 0.9

* Sun Jan 9 2005 Jeff Gilchrist <pbzip2@compression.ca> - 0.8.3-1
- Release 0.8.3

* Mon Nov 30 2004 Jeff Gilchrist <pbzip2@compression.ca> - 0.8.2-1
- Release 0.8.2

* Sat Nov 27 2004 Jeff Gilchrist <pbzip2@compression.ca> - 0.8.1-1
- Release 0.8.1

* Thu Oct 28 2004 Bryan Stillwell <bryan@bokeoa.com> - 0.8-1
- Initial packaging
