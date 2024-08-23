//
// Created by peiming on 11/13/19.
//
#include <llvm/ADT/StringRef.h>

#include <set>

#include "PointerAnalysis/Models/LanguageModel/DefaultLangModel/DefaultExtFunctions.h"

using namespace llvm;

const std::set<StringRef> xray::DefaultExtFunctions::THREAD_CREATIONS{
    "pthread_create"};