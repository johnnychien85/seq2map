#ifndef SPARSE_NODE_HPP
#define SPARSE_NODE_HPP

#include <boost/iterator/iterator_facade.hpp>
#include <seq2map/common.hpp>

namespace seq2map
{
    template<typename T, size_t dims>
    class NodeND
    {
    private:
        typedef NodeND<T, dims> NodeType;
        typedef NodeType* NodePtr;
        typedef const NodeType* ConstNodePtr;

    public:
        /**
         * The container is for internal use only to hold linked nodes.
         * The idea of container will be extended by dimensions.
         */
        class Container : public Indexed
        {
        public:
            Container(size_t idx) : Indexed(idx), m_head(NULL), m_tail(NULL) {}
            virtual String ToString() const = 0;

        public: // STL-compliance members
            virtual void insert(NodePtr node) = 0;
            virtual void  erase(NodePtr node) = 0;
            virtual bool  empty() const { return m_head == NULL; }
            virtual void  clear() { while (!empty()) erase(m_tail); }

        protected:
            NodePtr m_head;
            NodePtr m_tail;
        };

    private:
        /**
         * Link in a specific dimension
         */
        struct Link
        {
            Link() : d(NULL), prev(NULL), next(NULL), indexed(false) {}

            Container* d;
            NodePtr prev;
            NodePtr next;
            bool indexed;
        };

        NodeND(const T& value) : m_value(value) {}

        virtual ~NodeND()
        {
            // E_TRACE << "destroying " << ToString();
        }

        String ToString() const
        {
            std::stringstream ss;

            ss << "(";
            for (size_t d = 0; d < dims; d++)
            {
                ss << m_links[d].d->GetIndex() << (d < dims - 1 ? "," : "");
            }
            ss << ")";

            ss << " [" << m_value.index << "]";

            return ss.str();
        }

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
            Dimension(size_t idx) : Container(idx) {}

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

        private: // some data types needed for indexing and search
            /**
             * Node used in indexed search phase.
             */
            struct INode
            {
                NodePtr node;
                INode(const NodePtr node = NULL) : node(node) {}

                bool operator< (const INode& rhs) const
                {
                    static const size_t d0 = dim < idepth ? (idepth + 1) : idepth;
                    const INode& lhs = *this;

                    assert(d0 <= dims);
                    assert(rhs.node != NULL);

                    for (size_t d = 0; d < d0; d++)
                    {
                        assert(lhs.node->m_links[d].d != NULL);
                        assert(rhs.node->m_links[d].d != NULL);

                        if (*(lhs.node->m_links[d].d) < *(rhs.node->m_links[d].d))
                        {
                            // E_TRACE << lhs.node->ToString() << " < " << rhs.node->ToString() << ", d = " << d;
                            return true;
                        }
                        else if (*(rhs.node->m_links[d].d) < *(lhs.node->m_links[d].d))
                        {
                            // E_TRACE << rhs.node->ToString() << " < " << lhs.node->ToString() << ", d = " << d;
                            return false;
                        }
                    }
                    // E_TRACE << lhs.node->ToString() << " not less than " << rhs.node->ToString() << ", d0 = " << d0;
                    return false;
                }

                bool operator== (const INode& rhs) const
                {
                    static const size_t d0 = dim < idepth ? (idepth + 1) : idepth;
                    const INode& lhs = *this;

                    assert(d0 <= dims);

                    for (size_t d = 0; d < d0; d++)
                    {
                        if (!(*(lhs.node->m_links[d].d) == *(rhs.node->m_links[d].d)))
                        {
                            return false;
                        }
                    }
                    return true;
                }

                String ToString() const
                {
                    static const size_t d0 = dim < idepth ? (idepth + 1) : idepth;
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
            };

            // node used in linear search phase
            struct LNode
            {
                NodePtr node;
                LNode(const NodePtr node = NULL) : node(node) {}

                bool operator< (const LNode& rhs) const
                {
                    static const size_t d0 = dim < idepth ? (idepth + 1) : idepth;
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
            };

            typedef std::map<INode, NodePtr> Indexer;
            Indexer m_imap;

        protected: // STL-compliance members
            virtual void insert(NodePtr node)
            {
                Link& link = node->m_links[dim];

                assert(link.d == this); // ownership assertion

                NodePtr prev = upper_bound(node);
                NodePtr next = prev == NULL ? m_head : prev->m_links[dim].next;

                if (prev == NULL)
                {
                    m_head = node;
                }
                else
                {
                    prev->m_links[dim].next = node;
                }

                if (next == NULL)
                {
                    m_tail = node;
                }
                else
                {
                    next->m_links[dim].prev = node;
                }

                link.prev = prev;
                link.next = next;

                INode inode(node);
                NodePtr& indexed = m_imap[inode];

                if (indexed == NULL || !(LNode(node) < LNode(indexed)))
                {
                    if (indexed != NULL)
                    {
                        indexed->m_links[dim].indexed = false;
                        // E_TRACE << inode.ToString() << " unindexed " << indexed->ToString() << " due to insertion of " << node->ToString();
                    }

                    indexed = node;
                    node->m_links[dim].indexed = true;

                    // E_TRACE << inode.ToString() << " indexed " << node->ToString() << " due to node insertion (n=" << m_imap.size() << ")";
                    // E_TRACE << (prev ? prev->ToString() : "x") << " <- " << node->ToString() << " -> " << (next ? next->ToString() : "x");
                }
            }

            virtual void erase(NodePtr node)
            {
                Link& link = node->m_links[dim];

                if (link.prev != NULL) link.prev->m_links[dim].next = link.next;
                if (link.next != NULL) link.next->m_links[dim].prev = link.prev;

                if (node == m_head) m_head = link.next;
                if (node == m_tail) m_tail = link.prev;

                INode inode(node);
                Indexer::iterator itr = m_imap.find(inode);

                if (itr == m_imap.end())
                {
                    E_ERROR << "this can't be!!";
                    E_ERROR << "I am " << node->ToString();

                    E_ERROR << "dump of imap (n=" << m_imap.size() << ")";

                    for (Indexer::iterator it = m_imap.begin(); it != m_imap.end(); it++)
                    {
                        E_ERROR << (it->second == NULL ? "??" : it->second->ToString());
                    }

                    E_ERROR << "dump of dimension " << dim << ":";

                    for (NodePtr p = m_head; p != m_tail; p = p->m_links[dim].next)
                    {
                        E_ERROR << p->ToString();
                    }

                    
                    E_ERROR << "hell NOOOOOOOOOOOOOOOOO";
                }

                assert(itr != m_imap.end());
                assert(itr->second != NULL);

                if (link.indexed)
                {
                    assert(itr->second == node);

                    if (link.prev != NULL && (INode(link.prev) == inode))
                    {
                        NodeType& prev = *link.prev;

                        assert( prev.m_links[dim].d != NULL);
                        assert(!prev.m_links[dim].indexed);

                        prev.m_links[dim].indexed = true;
                        itr->second = link.prev;

                        // E_TRACE << inode.ToString() << " indexed " << prev.ToString() << " due to removal of " << node->ToString();
                    }
                    else
                    {
                        // E_TRACE << inode.ToString() << " erasing " << node->ToString() << " (n=" << m_imap.size() << ")";
                        m_imap.erase(itr);

                        //size_t erased = m_imap.erase(inode);
                        //assert(erased == 1);

                        // E_TRACE << inode.ToString() << " unindexed " << node->ToString() << " due to node removal (n=" << m_imap.size() << ")";
                    }

                    link.indexed = false;
                }
                else
                {
                    assert(itr->second != node);
                }

                // E_TRACE << "node " << node->ToString() << " erased in dimension " << dim;
            }

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
                    return m_tail;
                }

                // try first element - O(1) complexity
                if (inode < INode(m_head))
                {
                    // E_TRACE << "insert before head " << m_head->ToString();
                    return NULL;
                }

                // perform indexed linear search - O(lgN+M) complexity in worst case
                Indexer::const_iterator itr = m_imap.upper_bound(inode);

                NodePtr rb = (itr == m_imap.cend())   ? m_tail : itr->second;
                NodePtr lb = (itr == m_imap.cbegin()) ? m_head : (--itr)->second;

                // E_TRACE << "bound : " << lb->ToString() << " - " << rb->ToString();

                if (inode < INode(rb))
                {
                    // E_TRACE << "insert after left bound";
                    return lb;
                }

                // do in-dimension backward linear search
                for (NodePtr prev = rb; prev != lb; prev = prev->m_links[dim].prev)
                {
                    // E_TRACE << "comparing " << prev->ToString();
                    if (LNode(prev) < lnode) return prev;
                }

                // E_TRACE << "insert after left bound after linear search";
                return lb;
            }

        public: // STL-compliance members
            typedef T value_type;

            template<typename node_ptr, typename value_type>
            class Iterator
            : public boost::iterator_facade<Iterator<node_ptr, value_type>, NodeType, boost::bidirectional_traversal_tag, T&>
            {
            public:
                Iterator() {}
                explicit Iterator(node_ptr node) : m_node(node) {}
                bool operator==(Iterator const& itr) const { return equal(itr); }

                inline operator bool() const { return m_node != NULL; }

            protected:
                friend class boost::iterator_core_access;
                friend class Dimension;

                bool equal(Iterator const& itr) const { return m_node == itr.m_node; }
                value_type& dereference() const { return m_node->value; }
                virtual void increment()    { m_node = (m_node == NULL) ? NULL : m_node->m_links[dim].next; }
                virtual void decreasement() { m_node = (m_node == NULL) ? NULL : m_node->m_links[dim].prev; }

                node_ptr m_node;
            }; // end of Iterator

            typedef Iterator<NodePtr, T> iterator;
            typedef Iterator<ConstNodePtr, T const> const_iterator;

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
        Map3() {}

        virtual ~Map3()
        {
            m_dim0.clear();
        }

        D0& Dim0(size_t i) { return Dim<D0, S0>(m_dim0, i); }
        D1& Dim1(size_t j) { return Dim<D1, S1>(m_dim1, j); }
        D2& Dim2(size_t k) { return Dim<D2, S2>(m_dim2, k); }

        inline T& Insert(size_t i, size_t j, size_t k, const T& value)
        {
            typedef NodeND<T, 3>::Container* dnType;
            dnType d12[2];
            d12[0] = static_cast<dnType>(&Dim1(j));
            d12[1] = static_cast<dnType>(&Dim2(k));

            return Dim0(i).Insert(value, d12);
        }

    private:
        typedef std::map<size_t, D0> S0;
        typedef std::map<size_t, D1> S1;
        typedef std::map<size_t, D2> S2;

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
