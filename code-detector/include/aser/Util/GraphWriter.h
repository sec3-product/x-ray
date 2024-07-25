#ifndef ASER_PTA_GRAPHWRITER_H
#define ASER_PTA_GRAPHWRITER_H

// a little modification on llvm build-in callgraph writer

#include <llvm/Support/GraphWriter.h>

namespace aser {

bool DisplayGraph(llvm::StringRef Filename, bool wait = true,
                  llvm::GraphProgram::Name program = llvm::GraphProgram::DOT);

template <typename GraphType>
class GraphWriter {
    llvm::raw_ostream &O;
    const GraphType &G;

    using DOTTraits = llvm::DOTGraphTraits<GraphType>;
    using GTraits = llvm::GraphTraits<GraphType>;
    using NodeRef = typename GTraits::NodeRef;
    using node_iterator = typename GTraits::nodes_iterator;
    using child_iterator = typename GTraits::ChildIteratorType;
    using edge_iterator = typename GTraits::ChildEdgeIteratorType;
    DOTTraits DTraits;

    static_assert(std::is_pointer<NodeRef>::value,
                  "FIXME: Currently GraphWriter requires the NodeRef type to be "
                  "a pointer.\nThe pointer usage should be moved to "
                  "DOTGraphTraits, and removed from GraphWriter itself.");

    // Writes the edge labels of the node to O and returns true if there are any
    // edge labels not equal to the empty string "".
    bool getEdgeSourceLabels(llvm::raw_ostream &O, NodeRef Node) {
        edge_iterator EI = GTraits::child_edge_begin(Node);
        edge_iterator EE = GTraits::child_edge_end(Node);
        bool hasEdgeSourceLabels = false;

        for (unsigned i = 0; EI != EE && i != 64; ++EI, ++i) {
            std::string label = DTraits.getEdgeSourceLabel(Node, EI);

            if (label.empty()) continue;

            hasEdgeSourceLabels = true;

            if (i) O << "|";

            O << "<s" << i << ">" << llvm::DOT::EscapeString(label);
        }

        if (EI != EE && hasEdgeSourceLabels) O << "|<s64>truncated...";

        return hasEdgeSourceLabels;
    }

public:
    GraphWriter(llvm::raw_ostream &o, const GraphType &g, bool SN) : O(o), G(g) { DTraits = DOTTraits(SN); }

    void writeGraph(const std::string &Title = "") {
        // Output the header for the callgraph...
        writeHeader(Title);

        // Emit all of the nodes in the callgraph...
        writeNodes();

        // Output any customizations on the callgraph
        llvm::DOTGraphTraits<GraphType>::addCustomGraphFeatures(G, *this);

        // Output the end of the callgraph
        writeFooter();
    }

    void writeHeader(const std::string &Title) {
        std::string GraphName = DTraits.getGraphName(G);

        if (!Title.empty())
            O << "digraph \"" << llvm::DOT::EscapeString(Title) << "\" {\n";
        else if (!GraphName.empty())
            O << "digraph \"" << llvm::DOT::EscapeString(GraphName) << "\" {\n";
        else
            O << "digraph unnamed {\n";

        if (DTraits.renderGraphFromBottomUp()) O << "\trankdir=\"BT\";\n";

        if (!Title.empty())
            O << "\tlabel=\"" << llvm::DOT::EscapeString(Title) << "\";\n";
        else if (!GraphName.empty())
            O << "\tlabel=\"" << llvm::DOT::EscapeString(GraphName) << "\";\n";
        O << DTraits.getGraphProperties(G);
        O << "\n";
    }

    void writeFooter() {
        // Finish off the callgraph
        O << "}\n";
    }

    void writeNodes() {
        // Loop over the callgraph, printing it out...
        for (const auto Node : llvm::nodes<GraphType>(G))
            if (!isNodeHidden(Node)) writeNode(Node);
    }

    bool isNodeHidden(NodeRef Node) { return DTraits.isNodeHidden(Node, G); }

    void writeNode(NodeRef Node) {
        std::string NodeAttributes = DTraits.getNodeAttributes(Node, G);

        O << "\tNode" << static_cast<const void *>(Node) << " [shape=record,";
        if (!NodeAttributes.empty()) O << NodeAttributes << ",";
        O << "label=\"{";

        if (!DTraits.renderGraphFromBottomUp()) {
            O << llvm::DOT::EscapeString(DTraits.getNodeLabel(Node, G));

            // If we should include the address of the node in the label, do so
            // now.
            std::string Id = DTraits.getNodeIdentifierLabel(Node, G);
            if (!Id.empty()) O << "|" << llvm::DOT::EscapeString(Id);

            std::string NodeDesc = DTraits.getNodeDescription(Node, G);
            if (!NodeDesc.empty()) O << "|" << llvm::DOT::EscapeString(NodeDesc);
        }

        std::string edgeSourceLabels;
        llvm::raw_string_ostream EdgeSourceLabels(edgeSourceLabels);
        bool hasEdgeSourceLabels = getEdgeSourceLabels(EdgeSourceLabels, Node);

        if (hasEdgeSourceLabels) {
            if (!DTraits.renderGraphFromBottomUp()) O << "|";

            O << "{" << EdgeSourceLabels.str() << "}";

            if (DTraits.renderGraphFromBottomUp()) O << "|";
        }

        if (DTraits.renderGraphFromBottomUp()) {
            O << llvm::DOT::EscapeString(DTraits.getNodeLabel(Node, G));

            // If we should include the address of the node in the label, do so
            // now.
            std::string Id = DTraits.getNodeIdentifierLabel(Node, G);
            if (!Id.empty()) O << "|" << llvm::DOT::EscapeString(Id);

            std::string NodeDesc = DTraits.getNodeDescription(Node, G);
            if (!NodeDesc.empty()) O << "|" << llvm::DOT::EscapeString(NodeDesc);
        }

        if (DTraits.hasEdgeDestLabels()) {
            O << "|{";

            unsigned i = 0, e = DTraits.numEdgeDestLabels(Node);
            for (; i != e && i != 64; ++i) {
                if (i) O << "|";
                O << "<d" << i << ">" << llvm::DOT::EscapeString(DTraits.getEdgeDestLabel(Node, i));
            }

            if (i != e) O << "|<d64>truncated...";
            O << "}";
        }

        O << "}\"];\n";  // Finish printing the "node" line

        // Output all of the edges now
        edge_iterator EI = GTraits::child_edge_begin(Node);
        edge_iterator EE = GTraits::child_edge_end(Node);
        for (unsigned i = 0; EI != EE && i != 64; ++EI, ++i)
            if (!DTraits.isNodeHidden(GTraits::edge_dest(*EI),G)) writeEdge(Node, i, EI);
        for (; EI != EE; ++EI)
            if (!DTraits.isNodeHidden(GTraits::edge_dest(*EI),G)) writeEdge(Node, 64, EI);
    }

    void writeEdge(NodeRef Node, unsigned edgeidx, edge_iterator EI) {
        if (NodeRef TargetNode = GTraits::edge_dest(*EI)) {
            int DestPort = -1;
            if (DTraits.getEdgeSourceLabel(Node, EI).empty()) edgeidx = -1;

            emitEdge(static_cast<const void *>(Node), edgeidx, static_cast<const void *>(TargetNode), DestPort,
                     DTraits.getEdgeAttributes(Node, EI, G));
        }
    }

    /// emitSimpleNode - Outputs a simple (non-record) node
    void emitSimpleNode(const void *ID, const std::string &Attr, const std::string &Label, unsigned NumEdgeSources = 0,
                        const std::vector<std::string> *EdgeSourceLabels = nullptr) {
        O << "\tNode" << ID << "[ ";
        if (!Attr.empty()) O << Attr << ",";
        O << " label =\"";
        if (NumEdgeSources) O << "{";
        O << llvm::DOT::EscapeString(Label);
        if (NumEdgeSources) {
            O << "|{";

            for (unsigned i = 0; i != NumEdgeSources; ++i) {
                if (i) O << "|";
                O << "<s" << i << ">";
                if (EdgeSourceLabels) O << llvm::DOT::EscapeString((*EdgeSourceLabels)[i]);
            }
            O << "}}";
        }
        O << "\"];\n";
    }

    /// emitEdge - Output an edge from a simple node into the graph...
    void emitEdge(const void *SrcNodeID, int SrcNodePort, const void *DestNodeID, int DestNodePort,
                  const std::string &Attrs) {
        if (SrcNodePort > 64) return;              // Eminating from truncated part?
        if (DestNodePort > 64) DestNodePort = 64;  // Targeting the truncated part?

        O << "\tNode" << SrcNodeID;
        if (SrcNodePort >= 0) O << ":s" << SrcNodePort;
        O << " -> Node" << DestNodeID;
        if (DestNodePort >= 0 && DTraits.hasEdgeDestLabels()) O << ":d" << DestNodePort;

        if (!Attrs.empty()) O << "[" << Attrs << "]";
        O << ";\n";
    }

    /// getOStream - Get the raw output stream into the graph file. Useful to
    /// write fancy things using addCustomGraphFeatures().
    llvm::raw_ostream &getOStream() { return O; }
};

template <typename GraphType>
llvm::raw_ostream &WriteGraph(llvm::raw_ostream &O, const GraphType &G, bool ShortNames = false,
                              const llvm::Twine &Title = "") {
    // Start the callgraph emission process...
    aser::GraphWriter<const GraphType> W(O, G, ShortNames);

    // Emit the callgraph.
    W.writeGraph(Title.str());

    return O;
}

template <class GraphType>
void WriteGraphToFile(const std::string &graphName, const GraphType &graph, bool simple = false) {
    std::string fileName = graphName + ".dot";
    llvm::outs() << "Writing '" << fileName << "'...";

    std::error_code ErrInfo;
    llvm::ToolOutputFile F(fileName, ErrInfo, llvm::sys::fs::OF_None);
    if (!ErrInfo) {
        // dump the ValueFlowGraph here
        aser::WriteGraph(F.os(), graph, simple);
        F.os().close();

        if (!F.os().has_error()) {
            llvm::outs() << "\n";
            F.keep();
            return;
        }
    }
    llvm::outs() << "  error opening file for writing!\n";
    F.os().clear_error();
}

}  // namespace aser

#endif
