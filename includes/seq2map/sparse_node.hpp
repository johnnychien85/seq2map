#ifndef SPARSE_NODE_HPP
#define SPARSE_NODE_HPP

#include <boost/iterator/iterator_facade.hpp>
#include <seq2map/common.hpp>

namespace seq2map
{
    /**
     *
     */
    template<typename T, size_t dims>
    class NodeND
    {
    private:
        typedef NodeND<T, dims> NodeType;
        typedef NodeType*       NodePtr;
        typedef const NodeType* ConstNodePtr;
        typedef T               ValueType;
        typedef const T         ConstValueType;
        typedef T&              ValueRef;
        typedef const T&        ConstValueRef;

    public:
        /**
         * The container is an abstract linked nodes for internal use only.
         * The implementation will be completed in Dimension class.
         */
        class Container : public Indexed
        {
        public:
            Container(size_t idx) : Indexed(idx), m_head(NULL), m_tail(NULL) {}
            virtual String ToString() const = 0;

        public:
            //
            // STL-compliance members
            //
            virtual void insert(NodePtr node) = 0;
            virtual void  erase(NodePtr node) = 0;
            virtual bool  empty() const { return m_head == NULL; }
            virtual void  clear() { while (!empty()) erase(m_tail); }

        protected:
            NodePtr m_head; ///< pointer to the first node
            NodePtr m_tail; ///< pointer to the last node
        };

    private:
        /**
         * Link in a specific dimension
         */
        struct Link
        {
            Link() : d(NULL), prev(NULL), next(NULL), indexed(false) {}

            Container* d; ///< containing list
            NodePtr prev; ///< pointer to the predeccessor
            NodePtr next; ///< pointer to the successor
            bool indexed; ///< flag indicating if the node is an indexed upper bound in the search tree
        };

        //
        // Constructor and destructor
        //

        /**
         *
         */
        NodeND(ConstValueRef value) : m_value(value) {}

        /**
         *
         */
        virtual ~NodeND()  { /* E_TRACE << "destroying " << ToString(); */ }

        //
        // Accessor
        //

        /**
         *
         */
        String ToString() const
        {
            std::stringstream ss;

            ss << "(";
            for (size_t d = 0; d < dims; d++)
            {
                bool more = d < dims - 1;
                if (m_links[d].d != NULL) ss << m_links[d].d->GetIndex();
                else ss << "?";
                
                if (more) ss << ",";
            }
            ss << ")";
            //ss << " [" << m_value.index << "]";

            return ss.str();
        }

        //
        // Data member
        //
        Link m_links[dims];
        T    m_value;

    public:
        /**
         * Dimension is the externally extensible class.
         */
        template<size_t dim, size_t idepth = 1>
        class Dimension : public Container
        {
        public:
            //
            // Constructor
            //
            Dimension(size_t idx) : Container(idx), m_nodes(0) {}

            //
            // Accessor
            //
            String ToString() const
            {
                std::stringstream ss;

                ss << "D" << dim << "[" << GetIndex() << "] : ";
                ss << "head -> ";
                for (const_iterator itr = cbegin(); itr; itr++)
                {
                    ss << itr.m_node->ToString() << (itr.m_node->m_links[dim].indexed ? "*" : "") << " -> ";
                }
                ss << "tail";

                return ss.str();
            }

        private:
            //
            // Data types needed for indexing and search.
            //

            /**
             * Node used in indexed tree search phase. The insertion of a node
             * involes two search levels, namely tree level and linear level.
             * At tree level the last possible successor is located, at where
             * a linear backward seach starts. Such a design is optimised for
             * increasing insertion index.
             */
            struct INode
            {
                //
                // Constructor
                //
                INode(const NodePtr node = NULL) : node(node) {}

                //
                // Comparisons
                //

                /**
                 *
                 */
                bool operator< (const INode& rhs) const
                {
                    const INode& lhs = *this;

                    // sanity checks
                    assert(lhs.node != NULL);
                    assert(rhs.node != NULL);
                    assert(d0 <= dims);

                    for (size_t d = 0; d < d0; d++)
                    {
                        // check if the nodes are still linked
                        assert(lhs.node->m_links[d].d != NULL);
                        assert(rhs.node->m_links[d].d != NULL);

                        const Container& dl = *(lhs.node->m_links[d].d);
                        const Container& dr = *(rhs.node->m_links[d].d);

                        if (dl < dr)
                        {
                            // E_TRACE << lhs.node->ToString() << " < " << rhs.node->ToString() << ", d = " << d;
                            return true;
                        }

                        if (dr < dl)
                        {
                            // E_TRACE << rhs.node->ToString() << " < " << lhs.node->ToString() << ", d = " << d;
                            return false;
                        }
                    }
                    // E_TRACE << lhs.node->ToString() << " not less than " << rhs.node->ToString() << ", d0 = " << d0;
                    return false;
                }

                /**
                 *
                 */
                bool operator== (const INode& rhs) const
                {
                    const INode& lhs = *this;

                    // sanity checks
                    assert(lhs.node != NULL);
                    assert(rhs.node != NULL);
                    assert(d0 <= dims);

                    for (size_t d = 0; d < d0; d++)
                    {
                        // check if the nodes are still linked
                        assert(lhs.node->m_links[d].d != NULL);
                        assert(rhs.node->m_links[d].d != NULL);

                        const Container& dl = *(lhs.node->m_links[d].d);
                        const Container& dr = *(rhs.node->m_links[d].d);

                        if (!(dl == dr))
                        {
                            return false;
                        }
                    }
                    return true;
                }

                /**
                 *
                 */
                String ToString() const
                {
                    std::stringstream ss;

                    ss << "(";
                    if (node != NULL)
                    {
                        for (size_t d = 0;  d < d0;   d++) ss << node->m_links[d].d->GetIndex() << (d < dims - 1 ? "," : "");
                        for (size_t d = d0; d < dims; d++) ss << "*" << (d < dims - 1 ? "," : "");
                    }
                    else
                    {
                        for (size_t d = 0;  d < d0;   d++) ss << "?" << (d < dims - 1 ? "," : "");
                        for (size_t d = d0; d < dims; d++) ss << "*" << (d < dims - 1 ? "," : "");
                    }
                    ss << ")";

                    return ss.str();
                }

                //
                // Data members
                //
                static const size_t d0 = dim < idepth ? (idepth + 1) : idepth;
                NodePtr node;
            }; // end of struct INode

            // node used in linear search phase
            struct LNode
            {
                LNode(const NodePtr node = NULL) : node(node) {}

                bool operator< (const LNode& rhs) const
                {
                    const LNode& lhs = *this;

                    for (size_t d = d0; d < dims; d++)
                    {
                        if ((*lhs.node->m_links[d].d) < (*rhs.node->m_links[d].d))
                        {
                            // E_TRACE << lhs.node->ToString() << " < " << rhs.node->ToString() << ", d = " << d;
                            return true;
                        }
                        else if ((*rhs.node->m_links[d].d) < (*lhs.node->m_links[d].d))
                        {
                            // E_TRACE << rhs.node->ToString() << " < " << lhs.node->ToString() << ", d = " << d;
                            return false;
                        }
                    }
                    // E_TRACE << lhs.node->ToString() << " not less than " << rhs.node->ToString() << ", d0 = " << d0;
                    return false;
                }

                //
                // Data members
                //
                static const size_t d0 = dim < idepth ? (idepth + 1) : idepth;
                NodePtr node;
            }; // end of struct LNode

            //
            // Helpers
            //
            static inline Link& GetLink(NodePtr node)
            {
                return node->m_links[dim];
            }

            /**
             * Attempt to update the index after the insertion of a new Node.
             */
            bool UpdateIndex(NodePtr node)
            {
                INode inode(node);
                Indexer::iterator itr = m_imap.find(node);

                NodePtr indexed = (itr == m_imap.end() ? NULL : itr->second);

                if (indexed && !(LNode(node) < LNode(indexed)))
                {
                    return false;
                }

                if (indexed != NULL)
                {
                    assert(indexed == GetLink(node).next);

                    m_imap.erase(itr);
                    GetLink(indexed).indexed = false;

                    // E_TRACE << inode.ToString() << " unindexed " << indexed->ToString() << " due to insertion of " << node->ToString();
                }

                m_imap.insert(Indexer::value_type(inode, node));
                GetLink(node).indexed = true;

                // E_TRACE << inode.ToString() << " indexed " << node->ToString() << " due to node insertion (n=" << m_imap.size() << ")";
                // E_TRACE << (prev ? prev->ToString() : "x") << " <- " << node->ToString() << " -> " << (next ? next->ToString() : "x");

                return true;
            }

            //
            // for debuging
            //
            void Dump()
            {
                std::stringstream ss;
                ss << "head <-> ";
                for (NodePtr p = m_head; p != NULL; p = GetLink(p).next)
                {
                    ss << p->ToString() << (GetLink(p).indexed ? " [x] " : "") << " <-> ";
                }
                ss << "tail";

                E_TRACE << ss.str();
            }

            void ShowNeighbours(NodePtr node)
            {
                std::stringstream ss;
                const Link& l = GetLink(node);

                ss << ((l.prev == NULL) ? "head" : l.prev->ToString()) << " <-> ";
                ss << node->ToString() << " <-> ";
                ss << ((l.next == NULL) ? "tail" : l.next->ToString());

                E_TRACE << ss.str();
            }

            //
            // Data member
            //
            typedef std::map<INode, NodePtr> Indexer;
            Indexer m_imap;

        protected:
            //
            // STL-compliance members
            //

            /**
             * Insert a node to the dimension.
             */
            virtual void insert(NodePtr node)
            {
                Link& link = GetLink(node);

                // ownership check
                assert(link.d == this);

                //NodePtr prev = upper_bound(node);
                //NodePtr next = prev == NULL ? m_head : prev->m_links[dim].next;

                //E_INFO << node->ToString();
                //Dump();

                link.next = upper_bound(node);
                link.prev = link.next == NULL ? m_tail : GetLink(link.next).prev;

                if (link.prev == NULL)
                {
                    m_head = node;
                }
                else
                {
                    GetLink(link.prev).next = node;
                }

                if (link.next == NULL)
                {
                    m_tail = node;
                }
                else
                {
                    GetLink(link.next).prev = node;
                }

                //Dump();

                if (link.prev != NULL)
                {
                    if (!(INode(link.prev) < INode(node) || !(LNode(node) < LNode(link.prev)))) Dump();
                    assert(INode(link.prev) < INode(node) || !(LNode(node) < LNode(link.prev)));
                }

                if (link.next != NULL)
                {
                    assert(INode(node) < INode(link.next) || !(LNode(link.next) < LNode(node)));
                }

                UpdateIndex(node);
                m_nodes++;
            }

            virtual void erase(NodePtr node)
            {
                Link& link = GetLink(node);

                if (link.prev != NULL) GetLink(link.prev).next = link.next;
                if (link.next != NULL) GetLink(link.next).prev = link.prev;

                if (node == m_head) m_head = link.next;
                if (node == m_tail) m_tail = link.prev;

                INode inode(node);
                Indexer::iterator itr = m_imap.find(inode);

                assert(itr != m_imap.end());
                assert(itr->second != NULL);

                NodePtr& indexed = itr->second;

                if (link.indexed)
                {
                    // E_TRACE << "updating index " << inode.ToString() << " for the removal of " << node->ToString();
                    // ShowNeighbours(node);

                    assert(indexed == node);

                    // remove index
                    // E_TRACE << inode.ToString() << " erasing " << node->ToString() << " (n=" << m_imap.size() << ")";
                    m_imap.erase(itr);
                    link.indexed = false;
                    // E_TRACE << inode.ToString() << " unindexed " << node->ToString() << " due to node removal (n=" << m_imap.size() << ")";

                    // try to transfer the index to the successor
                    if (link.next != NULL)
                    {
                        bool updated = UpdateIndex(link.next);

                        // if (updated)
                        // {
                        //   E_TRACE << inode.ToString() << " indexed " << link.next->ToString() << " due to removal of " << node->ToString() << "(n=" << m_imap.size() << ")";
                        // }
                    }
                }
                else
                {
                    assert(itr->second != node);
                }

                // E_TRACE << "node " << node->ToString() << " erased in dimension " << dim;
                m_nodes--;
            }

            /**
             * Search for the successor of a node for insertion into the dimension's linked list.
             */
            NodePtr upper_bound(NodePtr node) const
            {
                if (empty()) return NULL;

                // E_TRACE << node->ToString();
                INode inode(node);
                LNode lnode(node);

                // try last element first - O(1) complexity
                if (INode(m_tail) < inode)
                {
                    // E_TRACE << "insert after tail " << m_tail->ToString();
                    return NULL; // append to the end of list
                }

                // try first element - O(1) complexity
                if (inode < INode(m_head))
                {
                    // E_TRACE << "insert before head " << m_head->ToString();
                    return m_head; // prepend to the beginning of list
                }

                // perform indexed linear search - O(lgN+M) complexity in worst case
                Indexer::const_iterator itr = m_imap.upper_bound(inode);

                NodePtr rb = (itr == m_imap.cend())   ? m_tail : itr->second;
                NodePtr lb = (itr == m_imap.cbegin()) ? m_head : (--itr)->second;

                //E_TRACE << "bound : " << lb->ToString() << " - " << rb->ToString();

                // one node case
                if (lb == rb)
                {
                    return !(INode(lb) < inode) && lnode < LNode(lb) ? lb : NULL;
                }

                if (INode(lb) < inode)
                {
                    return rb;
                }

                // do in-dimension backward linear search
                for (NodePtr prev = rb; prev != GetLink(lb).prev; prev = GetLink(prev).prev)
                {
                    //E_INFO << prev->ToString();
                    if (!(inode < INode(prev)) && !(lnode < LNode(prev)))
                    {
                        //E_INFO "found";
                        return GetLink(prev).next;
                    }
                }

                // E_TRACE << "insert after left bound after linear search";
                return lb;
            }

        private:
            size_t m_nodes;

        public: // STL-compliance members
            typedef T value_type;

            template<typename node_ptr, typename value_type, typename ref_type>
            class Iterator
            : public boost::iterator_facade<Iterator<node_ptr, value_type, ref_type>, value_type, boost::bidirectional_traversal_tag, ref_type>
            {
            public:
                Iterator() {}
                explicit Iterator(node_ptr node) : m_node(node) {}
                bool operator==(Iterator const& itr) const { return equal(itr); }

                inline operator bool() const { return m_node != NULL; }

                template<size_t dim, class C>
                C& GetContainer()
                {
                    assert(dim < dims);
                    assert(m_node != NULL);
                    C* d = dynamic_cast<C*>(m_node->m_links[dim].d);
                    assert(d != NULL);

                    return *d;
                }

                template<size_t dim, class C>
                const C& GetContainer() const { return GetContainer<dim, const C>(); }

            protected:
                friend class boost::iterator_core_access;
                friend class Dimension;

                bool equal(Iterator const& itr) const { return m_node == itr.m_node; }
                ref_type dereference() const          { return m_node->m_value; }
                virtual void increment()    { m_node = (m_node == NULL) ? NULL : m_node->m_links[dim].next; }
                virtual void decreasement() { m_node = (m_node == NULL) ? NULL : m_node->m_links[dim].prev; }

                node_ptr m_node;
            }; // end of Iterator

            typedef Iterator<NodePtr, ValueType, ValueRef> iterator;
            typedef Iterator<ConstNodePtr, ConstValueType, ConstValueRef> const_iterator;

            inline size_t size() const { return m_nodes; }
            iterator begin() { return iterator(m_head); }
            iterator end()   { return iterator(m_tail); }

            const_iterator cbegin() const { return const_iterator(m_head); }
            const_iterator cend()   const { return const_iterator(m_tail); }

        }; // end of Dimension

        template<size_t idepth = 1>
        class DimensionZero : public Dimension<0, idepth>
        {
        public:
            DimensionZero(size_t idx) : Dimension(idx) {}

            virtual ~DimensionZero()
            {
                while (!empty())
                {
                    NodePtr node = m_tail;
                    Remove(node);
                }
            }

            T& Insert(const T& value, Container** dn)
            {
                NodePtr node = new NodeType(value);

                for (size_t d = 0; d < dims; d++)
                {
                    node->m_links[d].d = (d == 0) ? static_cast<Container*>(this) : dn[d - 1];
                }

                // E_TRACE << "inserting " << node->ToString();

                // link the node dimension-by-dimension
                for (size_t d = 0; d < dims; d++)
                {
                    // E_TRACE << node->m_links[d].d->ToString();
                    node->m_links[d].d->insert(node);
                    // E_TRACE << node->m_links[d].d->ToString();
                }

                return node->m_value;
            }

        protected:
            void Remove(NodePtr node)
            {
                // E_TRACE << "removing " << node->ToString();

                // unlink the node dimension-by-dimension
                for (size_t d = 0; d < dims; d++)
                {
                    // E_TRACE << node->m_links[d].d->ToString();
                    node->m_links[d].d->erase(node);
                    // E_TRACE << node->m_links[d].d->ToString();
                }

                for (size_t d = 0; d < dims; d++)
                {
                    node->m_links[d].d = NULL;
                }

                delete node;
            }
        };
    };

    template<
        typename T,
        class D0 = NodeND<T,3>::DimensionZero<1>,
        class D1 = NodeND<T,3>::Dimension<1,1>,
        class D2 = NodeND<T,3>::Dimension<2,1>
    >
    class Map3
    {
    public:
        typedef std::map<size_t, D0> S0;
        typedef std::map<size_t, D1> S1;
        typedef std::map<size_t, D2> S2;

        Map3() {}

        virtual ~Map3()
        {
            Clear();
        }

        D0& Dim0(size_t i) { return Dim<D0, S0>(m_dim0, i); }
        D1& Dim1(size_t j) { return Dim<D1, S1>(m_dim1, j); }
        D2& Dim2(size_t k) { return Dim<D2, S2>(m_dim2, k); }

        inline size_t GetSize0() const { return m_dim0.size(); }
        inline size_t GetSize1() const { return m_dim1.size(); }
        inline size_t GetSize2() const { return m_dim2.size(); }

        typename S0::const_iterator Begin0() const { return m_dim0.cbegin(); }
        typename S1::const_iterator Begin1() const { return m_dim1.cbegin(); }
        typename S2::const_iterator Begin2() const { return m_dim2.cbegin(); }

        typename S0::const_iterator End0() const { return m_dim0.cend(); }
        typename S1::const_iterator End1() const { return m_dim1.cend(); }
        typename S2::const_iterator End2() const { return m_dim2.cend(); }

        inline T& Insert(size_t i, size_t j, size_t k, const T& value)
        {
            typedef NodeND<T, 3>::Container* dnType;
            dnType d12[2];
            d12[0] = static_cast<dnType>(&Dim1(j));
            d12[1] = static_cast<dnType>(&Dim2(k));

            return Dim0(i).Insert(value, d12);
        }

        inline void Clear()
        {
            m_dim0.clear();
            m_dim1.clear();
            m_dim2.clear();
        }

    private:
        template<typename D, typename S>
        D& Dim(S& dims, size_t idx)
        {
            S::iterator itr = dims.find(idx);
            return itr != dims.end() ? itr->second :
                dims.insert(S::value_type(idx, D(idx))).first->second;
        }

        S0 m_dim0;
        S1 m_dim1;
        S2 m_dim2;
    };
}
#endif // SPARSE_NODE_HPP
